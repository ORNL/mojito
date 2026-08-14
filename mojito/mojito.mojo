from max.gpu.host import DeviceContext
from max.gpu.host.device_context import DefaultDeviceTypeEncoder
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from std.gpu import block_dim, block_idx, thread_idx
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder

from std.memory import stack_allocation, alloc
from std.collections import Optional
from max.algorithm import parallelize
from std.math import ceildiv, min
from max.runtime.asyncrt import parallelism_level

comptime TBSize = 512

# GPU-passable view of mojito array, avoids host-only fields
struct array_ref[
    dtype: DType,
    Nx: Int,
    Ny: Int = 1,
    Nz: Int = 1,
](DevicePassable, ImplicitlyCopyable):
    comptime _N = Self.Nx * Self.Ny * Self.Nz
    var _data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    comptime device_type = Self

    def __init__(
        out self,
        data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin],
    ):
        self._data = data

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "MojitoArrayView"

    @staticmethod
    def get_device_type_name() -> String:
        return "MojitoArrayView"

    # Getters and setters for GPU kernel array manipulation
    def __getitem__(self, x: Int) -> Scalar[Self.dtype]:
        return self._data[unsafe_offset=x]
    def __getitem__(self, x: Int, y: Int) -> Scalar[Self.dtype]:
        return self._data[unsafe_offset=x * Self.Ny + y]
    def __getitem__(self, x: Int, y: Int, z: Int) -> Scalar[Self.dtype]:
        return self._data[unsafe_offset=x * Self.Ny * Self.Nz + y * Self.Nz + z]

    def __setitem__(self, x: Int, value: Scalar[Self.dtype]):
        self._data[unsafe_offset=x] = value
    def __setitem__(self, x: Int, y: Int, value: Scalar[Self.dtype]):
        self._data[unsafe_offset=x * Self.Ny + y] = value
    def __setitem__(self, x: Int, y: Int, z: Int, value: Scalar[Self.dtype]):
        self._data[unsafe_offset=x * Self.Ny * Self.Nz + y * Self.Nz + z] = value


struct array[
    backend: String,
    dtype: DType,
    Nx: Int,
    Ny: Int = 1,
    Nz: Int = 1,
](DevicePassable, ImplicitlyCopyable):
    comptime _N = Self.Nx * Self.Ny * Self.Nz
    var _ctx : Optional[DeviceContext]
    var _data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var _on_host: Bool
    var _owned: Bool

    # GPU array constructor (no fill value)
    def __init__(
        out self,
        ctx: DeviceContext
    ) raises:
        self._ctx = ctx
        var buf = ctx.enqueue_create_buffer[Self.dtype](Self._N)
        self._data = buf.take_ptr()
        self._on_host = False
        self._owned = True

    # GPU array constructor (with fill value)
    def __init__(
        out self,
        ctx: DeviceContext,
        filler: Scalar[Self.dtype]
    ) raises:
        self._ctx = ctx
        var buf = ctx.enqueue_create_buffer[Self.dtype](Self._N)
        buf.enqueue_fill(filler)
        self._data = buf.take_ptr()
        self._on_host = False
        self._owned = True

    # CPU array constructor (no fill value)
    def __init__(out self) raises:
        self._ctx = None
        self._data = alloc[Scalar[Self.dtype]](Self._N)
        self._on_host = True
        self._owned = True

    # CPU array constructor (with fill value)
    def __init__(out self, filler: Scalar[Self.dtype]) raises:
        self._ctx = None
        self._data = alloc[Scalar[Self.dtype]](Self._N)
        for i in range(Self._N):
            self._data[unsafe_offset=i] = filler
        self._on_host = True
        self._owned = True

    # Internal constructor for building copy results (used by Mojito.copy_to_*)
    def __init__(
        out self,
        ctx: Optional[DeviceContext],
        data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin],
        on_host: Bool,
        _owned: Bool
    ):
        self._ctx = ctx
        self._data = data
        self._on_host = on_host
        self._owned = _owned

    # 1D indexing
    def __getitem__(ref self, x: Int) raises -> Scalar[Self.dtype]:
        return self._data[unsafe_offset=x]

    # 2D indexing
    def __getitem__(ref self, x: Int, y: Int) raises -> Scalar[Self.dtype]:
        return self._data[unsafe_offset=x * Self.Ny + y]

    # 3D indexing
    def __getitem__(ref self, x: Int, y: Int, z: Int) raises -> Scalar[Self.dtype]:
        return self._data[unsafe_offset=x * Self.Ny * Self.Nz + y * Self.Nz + z]

    # 1D setitem
    def __setitem__(mut self, x: Int, value: Scalar[Self.dtype]) raises:
        self._data[unsafe_offset=x] = value

    # 2D setitem
    def __setitem__(mut self, x: Int, y: Int, value: Scalar[Self.dtype]) raises:
        self._data[unsafe_offset=x * Self.Ny + y] = value

    # 3D setitem
    def __setitem__(mut self, x: Int, y: Int, z: Int, value: Scalar[Self.dtype]) raises:
        self._data[unsafe_offset=x * Self.Ny * Self.Nz + y * Self.Nz + z] = value

    # Move device buffer to host for GPU backend, does nothing in other cases
    def to_host(mut self) raises:
        if self._ctx and not self._on_host:
            var h_buff = self._ctx.value().enqueue_create_host_buffer[Self.dtype](Self._N)
            h_buff.enqueue_copy_from(self._data)
            self._data = h_buff.take_ptr()
            self._on_host = True

    # Move host buffer to device for GPU backend, does nothing in other cases
    def to_device(mut self) raises:
        if self._ctx and self._on_host:
            var d_buff = self._ctx.value().enqueue_create_buffer[Self.dtype](Self._N)
            d_buff.enqueue_copy_from(self._data)
            self._data = d_buff.take_ptr()
            self._on_host = False

    def __deinit__(deinit self):
        if self._owned:
            comptime if Self.backend == "cpu":
                self._data.unsafe_free()
            # GPU data cleanup: TODO ??

    # DevicePassable requirements
    comptime device_type = array_ref[Self.dtype, Self.Nx, Self.Ny, Self.Nz]

    def _view(self) -> array_ref[Self.dtype, Self.Nx, Self.Ny, Self.Nz]:
        return array_ref[Self.dtype, Self.Nx, Self.Ny, Self.Nz](self._data)

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self._view(), target)

    @staticmethod
    def get_type_name() -> String:
        return "MojitoArray"

    @staticmethod
    def get_device_type_name() -> String:
        return "MojitoArrayView"


struct Mojito[backend: String]():
    var _ctx: Optional[DeviceContext]

    def __init__(out self) raises:
        comptime if Self.backend == "gpu":
            self._ctx = DeviceContext()
        else:
            self._ctx = None

    def get_ctx(self) raises -> DeviceContext:
        comptime if Self.backend != "gpu":
            raise Error("DeviceContext is only available for GPU backend")
        return self._ctx.value()

    def empty[
        type: DType,
        Nx: Int,
        Ny: Int = 1,
        Nz: Int = 1
    ](mut self) raises -> array[Self.backend, type, Nx, Ny, Nz]:
        comptime if Self.backend == "gpu":
            return array[Self.backend, type, Nx, Ny, Nz](self._ctx.value())
        else:
            return array[Self.backend, type, Nx, Ny, Nz]()

    def zeros[
        type: DType,
        Nx: Int,
        Ny: Int = 1,
        Nz: Int = 1
    ](mut self) raises -> array[Self.backend, type, Nx, Ny, Nz]:
        comptime if Self.backend == "gpu":
            return array[Self.backend, type, Nx, Ny, Nz](self._ctx.value(), Scalar[type](0))
        else:
            return array[Self.backend, type, Nx, Ny, Nz](Scalar[type](0))

    def ones[
        type: DType,
        Nx: Int,
        Ny: Int = 1,
        Nz: Int = 1
    ](mut self) raises -> array[Self.backend, type, Nx, Ny, Nz]:
        comptime if Self.backend == "gpu":
            return array[Self.backend, type, Nx, Ny, Nz](self._ctx.value(), Scalar[type](1))
        else:
            return array[Self.backend, type, Nx, Ny, Nz](Scalar[type](1))

    def fill[
        type: DType,
        Nx: Int,
        Ny: Int = 1,
        Nz: Int = 1
    ](mut self, filler: Scalar[type]) raises -> array[Self.backend, type, Nx, Ny, Nz]:
        comptime if Self.backend == "gpu":
            return array[Self.backend, type, Nx, Ny, Nz](self._ctx.value(), Scalar[type](filler))
        else:
            return array[Self.backend, type, Nx, Ny, Nz](filler)

    # Synchronize DeviceContext for GPU backend
    def sync(mut self) raises:
        if self._ctx:
            self._ctx.value().synchronize()


    # Return a new array with the data in host memory, leaving src unchanged
    # GPU + src on device: allocates a new host buffer and copies the data
    # GPU + src already on host, or CPU: returns a shallow copy
    def copy_to_host[
        type: DType,
        Nx: Int,
        Ny: Int = 1,
        Nz: Int = 1
    ](self, src: array[Self.backend, type, Nx, Ny, Nz]) raises
      -> array[Self.backend, type, Nx, Ny, Nz]:
        comptime A = array[Self.backend, type, Nx, Ny, Nz]

        if Self.backend == "gpu" and not src._on_host:
            var h_buf = self._ctx.value().enqueue_create_host_buffer[type](A._N)
            h_buf.enqueue_copy_from(src._data)
            return A(src._ctx, h_buf.take_ptr(), True, True)
        else:
            return A(src._ctx, src._data, True, False)


    # Return a new array with the data in device memory, leaving src unchanged
    # GPU + src on host: allocates a new device buffer and copies the data
    # GPU + src already on device, or CPU: returns a shallow copy
    def copy_to_device[
        type: DType,
        Nx: Int,
        Ny: Int = 1,
        Nz: Int = 1
    ](self, src: array[Self.backend, type, Nx, Ny, Nz]) raises
      -> array[Self.backend, type, Nx, Ny, Nz]:
        comptime A = array[Self.backend, type, Nx, Ny, Nz]

        if Self.backend == "gpu" and src._on_host:
            var d_buf = self._ctx.value().enqueue_create_buffer[type](A._N)
            d_buf.enqueue_copy_from(src._data)
            return A(src._ctx, d_buf.take_ptr(), False, True)
        else:
            return A(src._ctx, src._data, src._on_host, False)

    # parallel_for overloads for 1, 2, and 3 arguments
    def parallel_for[
        Nx: Int,
        V1: DevicePassable,
        func: def(i: Int, v1: V1.device_type) thin -> None,
        num_threads: Int = TBSize,
    ](mut self, v1: V1) raises:
        comptime if Self.backend == "gpu":
            def kernel(v1: V1.device_type):
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                if i < Nx:
                    func(i, v1)

            self._ctx.value().enqueue_function[kernel](
                v1,
                grid_dim = (ceildiv(Nx, num_threads)),
                block_dim = num_threads
            )
            self._ctx.value().synchronize()
        # CPU path:
        else:
            # func() must take a device_type, so we must convert each argument
            # to device_type even for the CPU path.
            # _to_device_type() mutates a pointer, so we must allocate it first
            var dv = stack_allocation[1, V1.device_type]()
            var enc = DefaultDeviceTypeEncoder()
            # _to_device_type() takes a void pointer, so we need to cast
            v1._to_device_type(enc, dv.unsafe_bitcast[NoneType]())

            def wrapper(i: Int) capturing -> None:
                func(i, dv[unsafe_offset=0])
            parallelize[wrapper](Nx)

    def parallel_for[
        Nx: Int,
        V1: DevicePassable,
        V2: DevicePassable,
        func: def(i: Int, v1: V1.device_type, v2: V2.device_type) thin -> None,
        num_threads: Int = TBSize,
    ](mut self, v1: V1, v2: V2) raises:
        comptime if Self.backend == "gpu":
            def kernel(v1: V1.device_type, v2: V2.device_type):
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                if i < Nx:
                    func(i, v1, v2)

            self._ctx.value().enqueue_function[kernel](
                v1, v2,
                grid_dim = (ceildiv(Nx, num_threads)),
                block_dim = num_threads
            )
            self._ctx.value().synchronize()
        # CPU path:
        else:
            var dv1 = stack_allocation[1, V1.device_type]()
            var dv2 = stack_allocation[1, V2.device_type]()
            var enc = DefaultDeviceTypeEncoder()

            v1._to_device_type(enc, dv1.unsafe_bitcast[NoneType]())
            v2._to_device_type(enc, dv2.unsafe_bitcast[NoneType]())

            def wrapper(i: Int) capturing -> None:
                func(i, dv1[unsafe_offset=0], dv2[unsafe_offset=0])
            parallelize[wrapper](Nx)

    def parallel_for[
        Nx: Int,
        V1: DevicePassable,
        V2: DevicePassable,
        V3: DevicePassable,
        func: def(i: Int, v1: V1.device_type, v2: V2.device_type, v3: V3.device_type) thin -> None,
        num_threads: Int = TBSize,
    ](mut self, v1: V1, v2: V2, v3: V3) raises:

        comptime if Self.backend == "gpu":
            def kernel(v1: V1.device_type, v2: V2.device_type, v3: V3.device_type):
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                if i < Nx:
                    func(i, v1, v2, v3)

            self._ctx.value().enqueue_function[kernel](
                v1, v2, v3,
                grid_dim = (ceildiv(Nx, num_threads)),
                block_dim = num_threads
            )
            self._ctx.value().synchronize()
        # CPU path:
        else:
            var dv1 = stack_allocation[1, V1.device_type]()
            var dv2 = stack_allocation[1, V2.device_type]()
            var dv3 = stack_allocation[1, V3.device_type]()
            var enc = DefaultDeviceTypeEncoder()

            v1._to_device_type(enc, dv1.unsafe_bitcast[NoneType]())
            v2._to_device_type(enc, dv2.unsafe_bitcast[NoneType]())
            v3._to_device_type(enc, dv3.unsafe_bitcast[NoneType]())

            def wrapper(i: Int) capturing -> None:
                func(i, dv1[unsafe_offset=0], dv2[unsafe_offset=0], dv3[unsafe_offset=0])
            parallelize[wrapper](Nx)


    def parallel_for[
        Nx: Int,
        V1: DevicePassable,
        V2: DevicePassable,
        V3: DevicePassable,
        V4: DevicePassable,
        func: def(i: Int, v1: V1.device_type, v2: V2.device_type, v3: V3.device_type, v4: V4.device_type) thin -> None,
        num_threads: Int = TBSize,
    ](mut self, v1: V1, v2: V2, v3: V3, v4: V4) raises:
        comptime if Self.backend == "gpu":
            def kernel(v1: V1.device_type, v2: V2.device_type, v3: V3.device_type, v4: V4.device_type):
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                if i < Nx:
                    func(i, v1, v2, v3, v4)
            self._ctx.value().enqueue_function[kernel](
                v1, v2, v3, v4,
                grid_dim = (ceildiv(Nx, num_threads)),
                block_dim = num_threads
            )
            self._ctx.value().synchronize()
        else:
            var dv1 = stack_allocation[1, V1.device_type]()
            var dv2 = stack_allocation[1, V2.device_type]()
            var dv3 = stack_allocation[1, V3.device_type]()
            var dv4 = stack_allocation[1, V4.device_type]()
            var enc = DefaultDeviceTypeEncoder()
            v1._to_device_type(enc, dv1.unsafe_bitcast[NoneType]())
            v2._to_device_type(enc, dv2.unsafe_bitcast[NoneType]())
            v3._to_device_type(enc, dv3.unsafe_bitcast[NoneType]())
            v4._to_device_type(enc, dv4.unsafe_bitcast[NoneType]())
            def wrapper(i: Int) capturing -> None:
                func(i, dv1[unsafe_offset=0], dv2[unsafe_offset=0], dv3[unsafe_offset=0], dv4[unsafe_offset=0])
            parallelize[wrapper](Nx)

    def parallel_for[
        Nx: Int,
        V1: DevicePassable,
        V2: DevicePassable,
        V3: DevicePassable,
        V4: DevicePassable,
        V5: DevicePassable,
        func: def(i: Int, v1: V1.device_type, v2: V2.device_type, v3: V3.device_type, v4: V4.device_type, v5: V5.device_type) thin -> None,
        num_threads: Int = TBSize,
    ](mut self, v1: V1, v2: V2, v3: V3, v4: V4, v5: V5) raises:
        comptime if Self.backend == "gpu":
            def kernel(v1: V1.device_type, v2: V2.device_type, v3: V3.device_type, v4: V4.device_type, v5: V5.device_type):
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                if i < Nx:
                    func(i, v1, v2, v3, v4, v5)
            self._ctx.value().enqueue_function[kernel](
                v1, v2, v3, v4, v5,
                grid_dim = (ceildiv(Nx, num_threads)),
                block_dim = num_threads
            )
            self._ctx.value().synchronize()
        else:
            var dv1 = stack_allocation[1, V1.device_type]()
            var dv2 = stack_allocation[1, V2.device_type]()
            var dv3 = stack_allocation[1, V3.device_type]()
            var dv4 = stack_allocation[1, V4.device_type]()
            var dv5 = stack_allocation[1, V5.device_type]()
            var enc = DefaultDeviceTypeEncoder()
            v1._to_device_type(enc, dv1.unsafe_bitcast[NoneType]())
            v2._to_device_type(enc, dv2.unsafe_bitcast[NoneType]())
            v3._to_device_type(enc, dv3.unsafe_bitcast[NoneType]())
            v4._to_device_type(enc, dv4.unsafe_bitcast[NoneType]())
            v5._to_device_type(enc, dv5.unsafe_bitcast[NoneType]())
            def wrapper(i: Int) capturing -> None:
                func(i, dv1[unsafe_offset=0], dv2[unsafe_offset=0], dv3[unsafe_offset=0], dv4[unsafe_offset=0], dv5[unsafe_offset=0])
            parallelize[wrapper](Nx)

    # 3D overloads
    # iz (fastest in row-major) = thread_idx.x
    def parallel_for[
        Nx: Int, Ny: Int, Nz: Int,
        V1: DevicePassable,
        func: def(ix: Int, iy: Int, iz: Int, v1: V1.device_type) thin -> None,
        num_threads: Int = TBSize,
    ](mut self, v1: V1) raises:
        comptime if Self.backend == "gpu":
            def kernel(v1: V1.device_type):
                var iz = Int(block_idx.x * block_dim.x + thread_idx.x)
                var iy = Int(block_idx.y * block_dim.y + thread_idx.y)
                var ix = Int(block_idx.z * block_dim.z + thread_idx.z)
                if ix < Nx and iy < Ny and iz < Nz:
                    func(ix, iy, iz, v1)
            self._ctx.value().enqueue_function[kernel](
                v1,
                grid_dim  = (ceildiv(Nz, num_threads), Ny, Nx),
                block_dim = (num_threads, 1, 1)
            )
            self._ctx.value().synchronize()
        else:
            var dv1 = stack_allocation[1, V1.device_type]()
            var enc = DefaultDeviceTypeEncoder()
            v1._to_device_type(enc, dv1.unsafe_bitcast[NoneType]())
            def wrapper(ix: Int) capturing -> None:
                for iy in range(Ny):
                    for iz in range(Nz):
                        func(ix, iy, iz, dv1[unsafe_offset=0])
            parallelize[wrapper](Nx)

    def parallel_for[
        Nx: Int, Ny: Int, Nz: Int,
        V1: DevicePassable,
        V2: DevicePassable,
        func: def(ix: Int, iy: Int, iz: Int, v1: V1.device_type, v2: V2.device_type) thin -> None,
        num_threads: Int = TBSize,
    ](mut self, v1: V1, v2: V2) raises:
        comptime if Self.backend == "gpu":
            def kernel(v1: V1.device_type, v2: V2.device_type):
                var iz = Int(block_idx.x * block_dim.x + thread_idx.x)
                var iy = Int(block_idx.y * block_dim.y + thread_idx.y)
                var ix = Int(block_idx.z * block_dim.z + thread_idx.z)
                if ix < Nx and iy < Ny and iz < Nz:
                    func(ix, iy, iz, v1, v2)
            self._ctx.value().enqueue_function[kernel](
                v1, v2,
                grid_dim  = (ceildiv(Nz, num_threads), Ny, Nx),
                block_dim = (num_threads, 1, 1)
            )
            self._ctx.value().synchronize()
        else:
            var dv1 = stack_allocation[1, V1.device_type]()
            var dv2 = stack_allocation[1, V2.device_type]()
            var enc = DefaultDeviceTypeEncoder()
            v1._to_device_type(enc, dv1.unsafe_bitcast[NoneType]())
            v2._to_device_type(enc, dv2.unsafe_bitcast[NoneType]())
            def wrapper(ix: Int) capturing -> None:
                for iy in range(Ny):
                    for iz in range(Nz):
                        func(ix, iy, iz, dv1[unsafe_offset=0], dv2[unsafe_offset=0])
            parallelize[wrapper](Nx)


    def parallel_reduce[
        N: Int,
        V1: DevicePassable,
        V2: DevicePassable,
        dtype: DType,
        func: def(i: Int, v1: V1.device_type, v2: V2.device_type) thin -> Scalar[dtype],
        num_threads: Int = TBSize,
    ](mut self, v1: V1, v2: V2) raises -> Scalar[dtype]:
        comptime num_blocks = ceildiv(N, num_threads)
        var res: Scalar[dtype] = 0

        comptime if Self.backend == "gpu":
            var partial = self._ctx.value().enqueue_create_buffer[dtype](num_blocks)

            def kernel(
                v1: V1.device_type,
                v2: V2.device_type,
                partial: Pointer[Scalar[dtype], MutAnyOrigin]
            ):
                var shared = stack_allocation[
                    num_threads,
                    Scalar[dtype],
                    address_space=AddressSpace.SHARED,
                ]()
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                var i_local = Int(thread_idx.x)
                if i < N:
                    shared[unsafe_offset=thread_idx.x] = func(i, v1, v2)
                else:
                    shared[unsafe_offset=thread_idx.x] = 0
                barrier()

                var offset = num_threads // 2
                while offset > 0:
                    if i_local < offset:
                        shared[unsafe_offset=i_local] += shared[unsafe_offset=i_local + offset]
                    barrier()
                    offset >>= 1

                if i_local == 0:
                    partial[unsafe_offset=block_idx.x] = shared[unsafe_offset=0]

            self._ctx.value().enqueue_function[kernel](
                v1, v2, partial,
                grid_dim=num_blocks,
                block_dim=num_threads
            )
            self._ctx.value().synchronize()

            with partial.map_to_host() as h_partial:
                for i in range(num_blocks):
                    res += h_partial[i]
        # CPU path:
        else:
            var dv1 = stack_allocation[1, V1.device_type]()
            var dv2 = stack_allocation[1, V2.device_type]()
            var enc = DefaultDeviceTypeEncoder()
            v1._to_device_type(enc, dv1.unsafe_bitcast[NoneType]())
            v2._to_device_type(enc, dv2.unsafe_bitcast[NoneType]())

            var num_workers = parallelism_level(self._ctx)
            var chunk = ceildiv(N, num_workers)
            var partials = List[Scalar[dtype]](length=num_workers, fill=0)

            def worker(tid: Int) capturing -> None:
                var start = tid * chunk
                var end = min(start + chunk, N)
                var s: Scalar[dtype] = 0
                for i in range(start, end):
                    s += func(i, dv1[unsafe_offset=0], dv2[unsafe_offset=0])
                partials[tid] = s

            parallelize[worker](num_workers)

            for i in range(num_workers):
                res += partials[i]

        return res
