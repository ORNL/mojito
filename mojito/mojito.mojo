# mojito v2: performance-portable arrays and parallel dispatch for Mojo.
#
# Design notes:
# - Arrays own refcounted DeviceBuffer/HostBuffer storage selected at
#   comptime by target; destruction is stream-ordered and automatic.
# - Kernel bodies are closures that capture array views BY VALUE (`var`
#   captures — `imm` captures are borrows of host memory and do not
#   survive the trip to a GPU).
# - Everything enqueued on a context is asynchronous; anything that
#   returns a value to the host blocks. Use fence() to await.
# - create_mirror may alias (and says so); deep_copy always copies
#   (and is a no-op on the same allocation).

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from max.gpu.sync import barrier
from max.runtime.asyncrt import parallelism_level

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.gpu import block_dim, block_idx, thread_idx, global_idx
from std.gpu.host.info import is_cpu, is_gpu, is_valid_target
from std.math import ceildiv, min, max
from std.memory import stack_allocation, MutOpaquePointer
from std.sys import has_accelerator, size_of
from std.utils.index import IndexList
from std.utils.numerics import max_finite, min_finite

comptime DefaultTarget: StaticString = "gpu" if has_accelerator() else "cpu"
comptime DefaultBlockSize = 256
# Minimum elements per CPU worker; PyTorch's TensorIterator grain size.
comptime DefaultGrainSize = 32768


# ===------------------------------------------------------------------=== #
# Execution policies
# ===------------------------------------------------------------------=== #


struct RangePolicy(ImplicitlyCopyable, TrivialRegisterPassable):
    """1D iteration range [begin, end) with tuning knobs."""

    var begin: Int
    var end: Int
    var block_size: Int
    var grain_size: Int

    def __init__(
        out self,
        end: Int,
        *,
        block_size: Int = DefaultBlockSize,
        grain_size: Int = DefaultGrainSize,
    ):
        self = Self(0, end, block_size=block_size, grain_size=grain_size)

    def __init__(
        out self,
        begin: Int,
        end: Int,
        *,
        block_size: Int = DefaultBlockSize,
        grain_size: Int = DefaultGrainSize,
    ):
        self.begin = begin
        self.end = end
        self.block_size = block_size
        self.grain_size = grain_size

    def size(self) -> Int:
        return self.end - self.begin


struct MDRangePolicy[rank: Int](ImplicitlyCopyable, TrivialRegisterPassable):
    """Multi-dimensional iteration space [0, ends) with tuning knobs.

    The innermost (last) dimension varies fastest, matching row-major
    array layout on both backends.
    """

    var ends: IndexList[Self.rank]
    var block_size: Int
    var grain_size: Int

    def __init__(
        out self,
        ends: IndexList[Self.rank],
        *,
        block_size: Int = DefaultBlockSize,
        grain_size: Int = DefaultGrainSize,
    ):
        self.ends = ends
        self.block_size = block_size
        self.grain_size = grain_size

    def size(self) -> Int:
        return self.ends.flattened_length()


# ===------------------------------------------------------------------=== #
# Reducers
# ===------------------------------------------------------------------=== #


trait ReduceOp:
    """A reduction monoid: an identity element and an associative join."""

    @staticmethod
    def init[dtype: DType]() -> Scalar[dtype]:
        ...

    @staticmethod
    def join[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
        ...


struct Sum(ReduceOp):
    @staticmethod
    def init[dtype: DType]() -> Scalar[dtype]:
        return 0

    @staticmethod
    def join[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
        return a + b


struct Prod(ReduceOp):
    @staticmethod
    def init[dtype: DType]() -> Scalar[dtype]:
        return 1

    @staticmethod
    def join[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
        return a * b


struct Min(ReduceOp):
    @staticmethod
    def init[dtype: DType]() -> Scalar[dtype]:
        return max_finite[dtype]()

    @staticmethod
    def join[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
        return min(a, b)


struct Max(ReduceOp):
    @staticmethod
    def init[dtype: DType]() -> Scalar[dtype]:
        return min_finite[dtype]()

    @staticmethod
    def join[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
        return max(a, b)


# ===------------------------------------------------------------------=== #
# Views
# ===------------------------------------------------------------------=== #


struct array_view[
    dtype: DType, rank: Int, origin: MutOrigin = MutUntrackedOrigin
](DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Non-owning, kernel-passable view of an array's data.

    Capture views in kernel bodies BY VALUE (`{var v}` capture lists).
    A view borrows its array (origin-tracked), so taking a view keeps
    the array alive through the view's last use; after an asynchronous
    launch the buffer's stream-ordered free protects the rest.
    """

    comptime device_type = Self
    var _data: Pointer[Scalar[Self.dtype], Self.origin]
    var _shape: IndexList[Self.rank]

    def __init__(
        out self,
        data: Pointer[Scalar[Self.dtype], Self.origin],
        shape: IndexList[Self.rank],
    ):
        self._data = data
        self._shape = shape

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "MojitoArrayView"

    def extent(self, d: Int) -> Int:
        return self._shape[d]

    def size(self) -> Int:
        return self._shape.flattened_length()

    # Linear indexing (any rank).
    def __getitem__(self, i: Int) -> Scalar[Self.dtype]:
        return self._data[unsafe_offset=i]

    def __setitem__(self, i: Int, value: Scalar[Self.dtype]):
        self._data[unsafe_offset=i] = value

    # 2D indexing.
    def __getitem__(self, i: Int, j: Int) -> Scalar[Self.dtype]:
        comptime assert Self.rank == 2, "2D indexing requires a rank-2 view"
        return self._data[unsafe_offset=i * self._shape[1] + j]

    def __setitem__(self, i: Int, j: Int, value: Scalar[Self.dtype]):
        comptime assert Self.rank == 2, "2D indexing requires a rank-2 view"
        self._data[unsafe_offset=i * self._shape[1] + j] = value

    # 3D indexing.
    def __getitem__(self, i: Int, j: Int, k: Int) -> Scalar[Self.dtype]:
        comptime assert Self.rank == 3, "3D indexing requires a rank-3 view"
        return self._data[
            unsafe_offset=(i * self._shape[1] + j) * self._shape[2] + k
        ]

    def __setitem__(self, i: Int, j: Int, k: Int, value: Scalar[Self.dtype]):
        comptime assert Self.rank == 3, "3D indexing requires a rank-3 view"
        self._data[
            unsafe_offset=(i * self._shape[1] + j) * self._shape[2] + k
        ] = value

    # SIMD access on the flattened index space.
    def load[width: Int](self, i: Int) -> SIMD[Self.dtype, width]:
        return self._data.unsafe_offset(i).unsafe_load[width=width]()

    def store[width: Int](self, i: Int, value: SIMD[Self.dtype, width]):
        self._data.unsafe_offset(i).unsafe_store(value)


# ===------------------------------------------------------------------=== #
# Arrays
# ===------------------------------------------------------------------=== #


struct array[target: StaticString, dtype: DType, rank: Int](ImplicitlyCopyable):
    """An owning, refcounted array with runtime extents.

    Copies share the underlying buffer (shallow, reference-counted);
    destruction is automatic and stream-ordered. Direct element access
    is only available where the data is host-accessible (`"cpu"` target
    arrays and mirrors) — for device arrays, use `create_mirror` +
    `deep_copy`.
    """

    comptime _BufferType: ImplicitlyCopyable = DeviceBuffer[
        Self.dtype
    ] if Self.target == "gpu" else HostBuffer[Self.dtype]

    var _buf: Self._BufferType
    var _shape: IndexList[Self.rank]

    def __init__(
        out self, var buf: Self._BufferType, shape: IndexList[Self.rank]
    ):
        self._buf = buf^
        self._shape = shape

    # The conditional _BufferType does not fold while `target` is generic;
    # these rebinds recover the concrete buffer type inside comptime-guarded
    # branches.
    def _dev(self) -> DeviceBuffer[Self.dtype]:
        comptime assert is_gpu[Self.target]()
        return rebind[DeviceBuffer[Self.dtype]](self._buf)

    def _host(self) -> HostBuffer[Self.dtype]:
        comptime assert is_cpu[Self.target]()
        return rebind[HostBuffer[Self.dtype]](self._buf)

    def size(self) -> Int:
        return self._shape.flattened_length()

    def extent(self, d: Int) -> Int:
        return self._shape[d]

    def view(
        mut self,
    ) -> array_view[Self.dtype, Self.rank, origin_of(self)]:
        comptime if is_gpu[Self.target]():
            return array_view[Self.dtype, Self.rank, origin_of(self)](
                self._dev()
                .unsafe_ptr()
                .unsafe_mut_cast[True]()
                .unsafe_origin_cast[origin_of(self)](),
                self._shape,
            )
        else:
            return array_view[Self.dtype, Self.rank, origin_of(self)](
                self._host()
                .unsafe_ptr()
                .unsafe_mut_cast[True]()
                .unsafe_origin_cast[origin_of(self)](),
                self._shape,
            )

    def fill(self, value: Scalar[Self.dtype]) raises:
        """Enqueues a fill of every element (asynchronous)."""
        comptime if is_gpu[Self.target]():
            self._dev().enqueue_fill(value)
        else:
            self._host().enqueue_fill(value)

    # Host element access: only for host-accessible arrays.
    def __getitem__(self, i: Int) raises -> Scalar[Self.dtype]:
        comptime assert is_cpu[Self.target](), (
            "direct indexing requires host-accessible data;"
            " use create_mirror + deep_copy"
        )
        var v = self
        return v.view()[i]

    def __setitem__(mut self, i: Int, value: Scalar[Self.dtype]) raises:
        comptime assert is_cpu[Self.target](), (
            "direct indexing requires host-accessible data;"
            " use create_mirror + deep_copy"
        )
        var v = self
        v.view()[i] = value

    def __getitem__(self, i: Int, j: Int) raises -> Scalar[Self.dtype]:
        comptime assert is_cpu[Self.target](), (
            "direct indexing requires host-accessible data;"
            " use create_mirror + deep_copy"
        )
        var v = self
        return v.view()[i, j]

    def __setitem__(mut self, i: Int, j: Int, value: Scalar[Self.dtype]) raises:
        comptime assert is_cpu[Self.target](), (
            "direct indexing requires host-accessible data;"
            " use create_mirror + deep_copy"
        )
        var v = self
        v.view()[i, j] = value

    def __getitem__(self, i: Int, j: Int, k: Int) raises -> Scalar[Self.dtype]:
        comptime assert is_cpu[Self.target](), (
            "direct indexing requires host-accessible data;"
            " use create_mirror + deep_copy"
        )
        var v = self
        return v.view()[i, j, k]

    def __setitem__(
        mut self, i: Int, j: Int, k: Int, value: Scalar[Self.dtype]
    ) raises:
        comptime assert is_cpu[Self.target](), (
            "direct indexing requires host-accessible data;"
            " use create_mirror + deep_copy"
        )
        var v = self
        v.view()[i, j, k] = value


# ===------------------------------------------------------------------=== #
# GPU kernel wrappers
# ===------------------------------------------------------------------=== #
# Callable structs (the pattern max.algorithm's elementwise uses) rather
# than nested closures: closure-state encoding of nested closures is not
# reliable for all capture compositions at the kernel boundary, while
# struct fields encode through the documented encode_fields path.


@fieldwise_init
struct _Kernel1D[
    FuncType: ImplicitlyCopyable & RegisterPassable & def(Int) -> None
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    var func: Self.FuncType
    var begin: Int
    var n: Int

    def __call__(self) capturing:
        var i = Int(global_idx.x)
        if i < self.n:
            self.func(self.begin + i)


@fieldwise_init
struct _Kernel2D[
    FuncType: ImplicitlyCopyable & RegisterPassable & def(Int, Int) -> None
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    var func: Self.FuncType
    var nx: Int
    var ny: Int

    def __call__(self) capturing:
        var j = Int(global_idx.x)
        var i = Int(global_idx.y)
        if i < self.nx and j < self.ny:
            self.func(i, j)


@fieldwise_init
struct _Kernel3D[
    FuncType: ImplicitlyCopyable & RegisterPassable & def(Int, Int, Int) -> None
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    var func: Self.FuncType
    var nx: Int
    var ny: Int
    var nz: Int

    def __call__(self) capturing:
        var k = Int(global_idx.x)
        var j = Int(global_idx.y)
        var i = Int(global_idx.z)
        if i < self.nx and j < self.ny and k < self.nz:
            self.func(i, j, k)


@fieldwise_init
struct _ReduceKernel[
    dtype: DType,
    FuncType: ImplicitlyCopyable & RegisterPassable & def(Int) -> Scalar[dtype],
    //,
    R: ReduceOp,
    block_size: Int,
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    var func: Self.FuncType
    var begin: Int
    var n: Int
    var partials: array_view[Self.dtype, 1]

    def __call__(self) capturing:
        var shared = stack_allocation[
            Self.block_size,
            Scalar[Self.dtype],
            address_space=AddressSpace.SHARED,
        ]()
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        var lane = Int(thread_idx.x)
        if i < self.n:
            shared[unsafe_offset=lane] = self.func(self.begin + i)
        else:
            shared[unsafe_offset=lane] = Self.R.init[Self.dtype]()
        barrier()

        var offset = Self.block_size // 2
        while offset > 0:
            if lane < offset:
                shared[unsafe_offset=lane] = Self.R.join(
                    shared[unsafe_offset=lane],
                    shared[unsafe_offset=lane + offset],
                )
            barrier()
            offset >>= 1

        if lane == 0:
            self.partials[Int(block_idx.x)] = shared[unsafe_offset=0]


# ===------------------------------------------------------------------=== #
# Mojito: context + dispatch
# ===------------------------------------------------------------------=== #


struct Mojito[target: StaticString = DefaultTarget]:
    """Execution context for one target ("cpu" or "gpu").

    GPU launches are asynchronous: parallel_for returns after enqueue and
    fence() awaits completion. CPU launches currently block on return (a
    conservative choice while async worker-closure lifetimes mature), so
    fence() is a cheap no-op there. parallel_reduce into a scalar always
    blocks.
    """

    var _ctx: DeviceContext

    def __init__(out self) raises:
        comptime assert is_valid_target[
            Self.target
        ](), "mojito target must be 'cpu' or 'gpu'"
        comptime if is_gpu[Self.target]():
            self._ctx = DeviceContext()
        else:
            self._ctx = DeviceContext(api="cpu")

    def ctx(self) -> DeviceContext:
        return self._ctx

    def fence(self) raises:
        """Blocks until all enqueued work on this context completes."""
        self._ctx.synchronize()

    # --- array creation (rank inferred from the number of extents) --- #

    def _make[
        dtype: DType, rank: Int
    ](self, shape: IndexList[rank]) raises -> array[Self.target, dtype, rank]:
        comptime A = array[Self.target, dtype, rank]
        comptime if is_gpu[Self.target]():
            return A(
                rebind[A._BufferType](
                    self._ctx.enqueue_create_buffer[dtype](
                        shape.flattened_length()
                    )
                ),
                shape,
            )
        else:
            return A(
                rebind[A._BufferType](
                    self._ctx.enqueue_create_host_buffer[dtype](
                        shape.flattened_length()
                    )
                ),
                shape,
            )

    def empty[
        dtype: DType
    ](self, nx: Int) raises -> array[Self.target, dtype, 1]:
        return self._make[dtype, 1](IndexList[1](nx))

    def empty[
        dtype: DType
    ](self, nx: Int, ny: Int) raises -> array[Self.target, dtype, 2]:
        return self._make[dtype, 2](IndexList[2](nx, ny))

    def empty[
        dtype: DType
    ](self, nx: Int, ny: Int, nz: Int) raises -> array[Self.target, dtype, 3]:
        return self._make[dtype, 3](IndexList[3](nx, ny, nz))

    def full[
        dtype: DType
    ](self, value: Scalar[dtype], nx: Int) raises -> array[
        Self.target, dtype, 1
    ]:
        var a = self.empty[dtype](nx)
        a.fill(value)
        return a

    def full[
        dtype: DType
    ](self, value: Scalar[dtype], nx: Int, ny: Int) raises -> array[
        Self.target, dtype, 2
    ]:
        var a = self.empty[dtype](nx, ny)
        a.fill(value)
        return a

    def full[
        dtype: DType
    ](self, value: Scalar[dtype], nx: Int, ny: Int, nz: Int) raises -> array[
        Self.target, dtype, 3
    ]:
        var a = self.empty[dtype](nx, ny, nz)
        a.fill(value)
        return a

    # --- mirrors and copies --- #

    def create_mirror[
        dtype: DType, rank: Int
    ](self, src: array[Self.target, dtype, rank]) raises -> array[
        "cpu", dtype, rank
    ]:
        """Returns a host-accessible array of the same shape.

        MAY ALIAS: when `src` is already host-accessible the mirror
        shares its storage. Use deep_copy to move data; it is a no-op
        when mirror and source alias.
        """
        comptime M = array["cpu", dtype, rank]
        comptime if is_gpu[Self.target]():
            return M(
                rebind[M._BufferType](
                    self._ctx.enqueue_create_host_buffer[dtype](src.size())
                ),
                src._shape,
            )
        else:
            return rebind[M](src)

    def deep_copy[
        dst_target: StaticString,
        src_target: StaticString,
        dtype: DType,
        rank: Int,
    ](
        self,
        dst: array[dst_target, dtype, rank],
        src: array[src_target, dtype, rank],
    ) raises:
        """Enqueues an element copy from src to dst (asynchronous).

        No-op when dst and src share the same allocation.
        """
        var d = dst
        var sr = src
        if d.view()._data == sr.view()._data:
            return
        comptime if is_gpu[dst_target]() and is_gpu[src_target]():
            dst._dev().enqueue_copy_from(src._dev())
        elif is_gpu[dst_target]() and is_cpu[src_target]():
            dst._dev().enqueue_copy_from(src._host())
        elif is_cpu[dst_target]() and is_gpu[src_target]():
            dst._host().enqueue_copy_from(src._dev())
        else:
            dst._host().enqueue_copy_from(src._host())

    # --- parallel_for: 1D --- #

    def parallel_for[
        FuncType: ImplicitlyCopyable & RegisterPassable & def(Int) -> None
    ](self, policy: RangePolicy, var func: FuncType) raises:
        """Executes func(i) for i in [policy.begin, policy.end).

        Asynchronous: returns after enqueue; call fence() to await.
        """
        var n = policy.size()
        if n <= 0:
            return

        comptime if is_gpu[Self.target]():
            var kernel = _Kernel1D(func^, policy.begin, n)
            self._ctx.enqueue_function(
                kernel,
                grid_dim=ceildiv(n, policy.block_size),
                block_dim=policy.block_size,
            )
        else:
            var num_workers = self._num_workers(n, policy.grain_size)
            var chunk = ceildiv(n, num_workers)
            var begin = policy.begin

            def worker(w: Int) {var chunk, var begin, var n, var func}:
                var lo = w * chunk
                var hi = min(lo + chunk, n)
                for i in range(lo, hi):
                    func(begin + i)

            self._ctx.enqueue_cpu_range(worker, count=num_workers)
            # TODO(v2): make CPU launches asynchronous once worker-closure
            # lifetime across an async enqueue_cpu_range is guaranteed; the
            # closure must outlive execution, so block here for now.
            self._ctx.synchronize()

    def parallel_for[
        FuncType: ImplicitlyCopyable & RegisterPassable & def(Int) -> None
    ](self, n: Int, var func: FuncType) raises:
        self.parallel_for(RangePolicy(n), func^)

    # --- parallel_for: 2D --- #

    def parallel_for[
        FuncType: ImplicitlyCopyable & RegisterPassable & def(Int, Int) -> None
    ](self, policy: MDRangePolicy[2], var func: FuncType) raises:
        var nx = policy.ends[0]
        var ny = policy.ends[1]
        if nx <= 0 or ny <= 0:
            return

        comptime if is_gpu[Self.target]():
            if nx > 65535:
                raise Error(
                    "MDRangePolicy[2]: extent 0 exceeds the 65535 grid limit"
                )

            var kernel = _Kernel2D(func^, nx, ny)
            self._ctx.enqueue_function(
                kernel,
                grid_dim=(ceildiv(ny, policy.block_size), nx),
                block_dim=(policy.block_size, 1),
            )
        else:
            # Parallelize the outer dimension, keep the inner serial
            # (contiguous) per worker.
            var num_workers = self._num_workers(nx * ny, policy.grain_size)
            num_workers = min(num_workers, nx)
            var chunk = ceildiv(nx, num_workers)

            def worker(w: Int) {var chunk, var nx, var ny, var func}:
                var lo = w * chunk
                var hi = min(lo + chunk, nx)
                for i in range(lo, hi):
                    for j in range(ny):
                        func(i, j)

            self._ctx.enqueue_cpu_range(worker, count=num_workers)
            self._ctx.synchronize()

    # --- parallel_for: 3D --- #

    def parallel_for[
        FuncType: ImplicitlyCopyable
        & RegisterPassable
        & def(Int, Int, Int) -> None
    ](self, policy: MDRangePolicy[3], var func: FuncType) raises:
        var nx = policy.ends[0]
        var ny = policy.ends[1]
        var nz = policy.ends[2]
        if nx <= 0 or ny <= 0 or nz <= 0:
            return

        comptime if is_gpu[Self.target]():
            if nx > 65535 or ny > 65535:
                raise Error(
                    "MDRangePolicy[3]: extents 0/1 exceed the 65535 grid limit"
                )

            var kernel = _Kernel3D(func^, nx, ny, nz)
            self._ctx.enqueue_function(
                kernel,
                grid_dim=(ceildiv(nz, policy.block_size), ny, nx),
                block_dim=(policy.block_size, 1, 1),
            )
        else:
            # Parallelize the two outer dimensions as rows, keep the
            # innermost serial (contiguous) per worker.
            var rows = nx * ny
            var num_workers = self._num_workers(rows * nz, policy.grain_size)
            num_workers = min(num_workers, rows)
            var chunk = ceildiv(rows, num_workers)

            def worker(w: Int) {var chunk, var nx, var ny, var nz, var func}:
                var rows_total = nx * ny
                var lo = w * chunk
                var hi = min(lo + chunk, rows_total)
                for r in range(lo, hi):
                    var i = r // ny
                    var j = r % ny
                    for k in range(nz):
                        func(i, j, k)

            self._ctx.enqueue_cpu_range(worker, count=num_workers)
            self._ctx.synchronize()

    # --- parallel_reduce --- #

    def parallel_reduce[
        R: ReduceOp,
        dtype: DType,
        FuncType: ImplicitlyCopyable
        & RegisterPassable
        & def(Int) -> Scalar[dtype],
        block_size: Int = DefaultBlockSize,
    ](self, policy: RangePolicy, var func: FuncType) raises -> Scalar[dtype]:
        """Reduces func(i) over [policy.begin, policy.end) with R.

        Returns the reduced scalar to the host, so this call BLOCKS.
        """
        comptime assert (
            block_size & (block_size - 1) == 0
        ), "parallel_reduce block_size must be a power of two"
        var n = policy.size()
        if n <= 0:
            return R.init[dtype]()

        comptime if is_gpu[Self.target]():
            var num_blocks = ceildiv(n, block_size)
            var partial = self._ctx.enqueue_create_buffer[dtype](num_blocks)
            var pview = array_view[dtype, 1](
                partial.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                IndexList[1](num_blocks),
            )
            var kernel = _ReduceKernel[R=R, block_size=block_size](
                func^, policy.begin, n, pview
            )
            self._ctx.enqueue_function(
                kernel, grid_dim=num_blocks, block_dim=block_size
            )

            # Read partials via an explicit host-buffer copy: on Metal,
            # map_to_host readback after a barrier kernel returns stale
            # data, while an enqueued copy is correctly ordered.
            var hpart = self._ctx.enqueue_create_host_buffer[dtype](num_blocks)
            hpart.enqueue_copy_from(partial)
            self._ctx.synchronize()
            var acc = R.init[dtype]()
            for b in range(num_blocks):
                acc = R.join(acc, hpart[b])
            return acc
        else:
            var num_workers = self._num_workers(n, policy.grain_size)
            var chunk = ceildiv(n, num_workers)
            var begin = policy.begin
            # Pad partial slots to a cache line to avoid false sharing.
            comptime slot_stride = max(1, 128 // size_of[Scalar[dtype]]())
            var partials = List[Scalar[dtype]](
                length=num_workers * slot_stride, fill=R.init[dtype]()
            )

            def worker(
                w: Int,
            ) {mut partials, imm func, imm chunk, imm begin, imm n}:
                var lo = w * chunk
                var hi = min(lo + chunk, n)
                var acc = R.init[dtype]()
                for i in range(lo, hi):
                    acc = R.join(acc, func(begin + i))
                partials[w * slot_stride] = acc

            # enqueue_cpu_range rather than sync_parallelize: the latter's
            # single-worker fast path runs inline without stream ordering,
            # racing against still-enqueued fills/copies.
            self._ctx.enqueue_cpu_range(worker, count=num_workers)
            self._ctx.synchronize()

            var acc = R.init[dtype]()
            for w in range(num_workers):
                acc = R.join(acc, partials[w * slot_stride])
            return acc

    def parallel_reduce[
        R: ReduceOp,
        dtype: DType,
        FuncType: ImplicitlyCopyable
        & RegisterPassable
        & def(Int) -> Scalar[dtype],
        block_size: Int = DefaultBlockSize,
    ](self, n: Int, var func: FuncType) raises -> Scalar[dtype]:
        return self.parallel_reduce[R, dtype, block_size=block_size](
            RangePolicy(n), func^
        )

    # --- internals --- #

    def _num_workers(self, n: Int, grain_size: Int) -> Int:
        return max(
            1,
            min(
                parallelism_level(self._ctx),
                ceildiv(n, grain_size),
                n,
            ),
        )
