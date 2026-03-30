from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx, barrier
from builtin.device_passable import DevicePassable

from memory import stack_allocation
from collections import Optional
from algorithm import parallelize, reduction
from math import ceildiv

# GPU-passable view of mojito array, avoids host-only fields
struct array_ref[
    dtype: DType,
    Nx: Int,
    Ny: Int = 1,
    Nz: Int = 1,
](DevicePassable, ImplicitlyCopyable):
    comptime _N = Self.Nx * Self.Ny * Self.Nz
    var _data: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    comptime device_type = Self

    fn __init__(
        out self,
        data: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
    ):
        self._data = data

    # Followed syntax from https://github.com/modular/modular/issues/6145
    fn _to_device_type[
        mut_origin: Origin[mut=True]
    ](
        self,
        target: UnsafePointer[NoneType, mut_origin]
    ):
        target.bitcast[Self]().init_pointee_copy(self)

    @staticmethod
    fn get_type_name() -> String:
        return "MojitoArrayView"

    @staticmethod
    fn get_device_type_name() -> String:
        return "MojitoArrayView"

    # Getters and setters for GPU kernel array manipulation
    fn __getitem__(self, x: Int) -> Scalar[Self.dtype]:
        return self._data[x]
    fn __getitem__(self, x: Int, y: Int) -> Scalar[Self.dtype]:
        return self._data[x * Self.Ny + y]
    fn __getitem__(self, x: Int, y: Int, z: Int) -> Scalar[Self.dtype]:
        return self._data[x * Self.Ny * Self.Nz + y * Self.Nz + z]

    fn __setitem__(self, x: Int, value: Scalar[Self.dtype]):
        self._data[x] = value
    fn __setitem__(self, x: Int, y: Int, value: Scalar[Self.dtype]):
        self._data[x * Self.Ny + y] = value
    fn __setitem__(self, x: Int, y: Int, z: Int, value: Scalar[Self.dtype]):
        self._data[x * Self.Ny * Self.Nz + y * Self.Nz + z] = value


struct array[
    backend: String,
    dtype: DType,
    Nx: Int,
    Ny: Int,
    Nz: Int
](DevicePassable, ImplicitlyCopyable):
    comptime _N = Self.Nx * Self.Ny * Self.Nz
    var _ctx : Optional[DeviceContext]
    var _data: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var _on_host: Bool
    var _owned: Bool

    # GPU array constructor (no fill value)
    fn __init__(
        out self,
        ctx: DeviceContext
    ) raises:
        self._ctx = ctx
        var buf = ctx.enqueue_create_buffer[Self.dtype](Self._N)
        self._data = buf.take_ptr()
        self._on_host = False
        self._owned = True

    # GPU array constructor (with fill value)
    fn __init__(
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
    fn __init__(out self) raises:
        self._ctx = None
        var list = List[Scalar[Self.dtype]](unsafe_uninit_length=Self._N)
        self._data = list.steal_data()
        self._on_host = True
        self._owned = True

    # CPU array constructor (with fill value)
    fn __init__(out self, filler: Scalar[Self.dtype]) raises:
        self._ctx = None
        var list = List[Scalar[Self.dtype]](length=Self._N, fill=filler)
        self._data = list.steal_data()
        self._on_host = True
        self._owned = True

    # Internal constructor for building copy results (used by Mojito.copy_to_*)
    fn __init__(
        out self,
        ctx: Optional[DeviceContext],
        data: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        on_host: Bool,
        _owned: Bool
    ):
        self._ctx = ctx
        self._data = data
        self._on_host = on_host
        self._owned = _owned

    # 1D indexing
    fn __getitem__(ref self, x: Int) raises -> Scalar[Self.dtype]:
        return self._data[x]

    # 2D indexing
    fn __getitem__(ref self, x: Int, y: Int) raises -> Scalar[Self.dtype]:
        return self._data[x * Self.Ny + y]

    # 3D indexing
    fn __getitem__(ref self, x: Int, y: Int, z: Int) raises -> Scalar[Self.dtype]:
        return self._data[x * Self.Ny * Self.Nz + y * Self.Nz + z]

    # 1D setitem
    fn __setitem__(mut self, x: Int, value: Scalar[Self.dtype]) raises:
        self._data[x] = value

    # 2D setitem
    fn __setitem__(mut self, x: Int, y: Int, value: Scalar[Self.dtype]) raises:
        self._data[x * Self.Ny + y] = value

    # 3D setitem
    fn __setitem__(mut self, x: Int, y: Int, z: Int, value: Scalar[Self.dtype]) raises:
        self._data[x * Self.Ny * Self.Nz + y * Self.Nz + z] = value

    # Move device buffer to host for GPU backend, does nothing in other cases
    fn to_host(mut self) raises:
        if self._ctx and not self._on_host:
            var h_buff = self._ctx.value().enqueue_create_host_buffer[Self.dtype](Self._N)
            h_buff.enqueue_copy_from(self._data)
            self._data = h_buff.take_ptr()
            self._on_host = True

    # Move host buffer to device for GPU backend, does nothing in other cases
    fn to_device(mut self) raises:
        if self._ctx and self._on_host:
            var d_buff = self._ctx.value().enqueue_create_buffer[Self.dtype](Self._N)
            d_buff.enqueue_copy_from(self._data)
            self._data = d_buff.take_ptr()
            self._on_host = False

    fn __del__(deinit self):
        if self._owned:
            @parameter
            if Self.backend == "cpu":
                self._data.free()
            # GPU data cleanup: TODO ??

    # DevicePassable requirements
    comptime device_type = array_ref[Self.dtype, Self.Nx, Self.Ny, Self.Nz]

    fn _view(self) -> array_ref[Self.dtype, Self.Nx, Self.Ny, Self.Nz]:
        return array_ref[Self.dtype, Self.Nx, Self.Ny, Self.Nz](self._data)

    fn _to_device_type[
        mut_origin: Origin[mut=True]
    ](
        self,
        target: UnsafePointer[NoneType, mut_origin]
    ):
        target.bitcast[array_ref[Self.dtype, Self.Nx, Self.Ny, Self.Nz]]()
            .init_pointee_copy(self._view())

    @staticmethod
    fn get_type_name() -> String:
        return "MojitoArray"

    @staticmethod
    fn get_device_type_name() -> String:
        return "MojitoArrayView"


struct Mojito[backend: String]():
    var _ctx: Optional[DeviceContext]

    fn __init__(out self) raises:
        @parameter
        if Self.backend == "gpu":
            self._ctx = DeviceContext()
        else:
            self._ctx = None

    fn get_ctx(self) raises -> DeviceContext:
        @parameter
        if Self.backend != "gpu":
            raise Error("DeviceContext is only available for GPU backend")
        return self._ctx.value()

    fn empty[
        type: DType,
        Nx: Int,
        Ny: Int = 1,
        Nz: Int = 1
    ](mut self) raises -> array[Self.backend, type, Nx, Ny, Nz]:
        @parameter
        if Self.backend == "gpu":
            return array[Self.backend, type, Nx, Ny, Nz](self._ctx.value())
        else:
            return array[Self.backend, type, Nx, Ny, Nz]()

    fn zeros[
        type: DType,
        Nx: Int,
        Ny: Int = 1,
        Nz: Int = 1
    ](mut self) raises -> array[Self.backend, type, Nx, Ny, Nz]:
        @parameter
        if Self.backend == "gpu":
            return array[Self.backend, type, Nx, Ny, Nz](self._ctx.value(), Scalar[type](0))
        else:
            return array[Self.backend, type, Nx, Ny, Nz](Scalar[type](0))

    fn ones[
        type: DType,
        Nx: Int,
        Ny: Int = 1,
        Nz: Int = 1
    ](mut self) raises -> array[Self.backend, type, Nx, Ny, Nz]:
        @parameter
        if Self.backend == "gpu":
            return array[Self.backend, type, Nx, Ny, Nz](self._ctx.value(), Scalar[type](1))
        else:
            return array[Self.backend, type, Nx, Ny, Nz](Scalar[type](1))

    fn fill[
        type: DType,
        Nx: Int,
        Ny: Int = 1,
        Nz: Int = 1
    ](mut self, filler: Scalar[type]) raises -> array[Self.backend, type, Nx, Ny, Nz]:
        @parameter
        if Self.backend == "gpu":
            return array[Self.backend, type, Nx, Ny, Nz](self._ctx.value(), Scalar[type](filler))
        else:
            return array[Self.backend, type, Nx, Ny, Nz](filler)

    # Synchronize DeviceContext for GPU backend
    fn sync(mut self) raises:
        if self._ctx:
            self._ctx.value().synchronize()


    # Return a new array with the data in host memory, leaving src unchanged
    # GPU + src on device: allocates a new host buffer and copies the data
    # GPU + src already on host, or CPU: returns a shallow copy
    fn copy_to_host[
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
    fn copy_to_device[
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
    fn parallel_for[
        N: Int,
        V1: DevicePassable,
        func: fn(i: Int, v1: V1.device_type) -> None,
    ](mut self, v1: V1) raises:
        @parameter
        if Self.backend == "gpu":
            fn kernel(v1: V1.device_type):
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                if i < N:
                    func(i, v1)

            self._ctx.value().enqueue_function[kernel, kernel](
                v1,
                grid_dim=ceildiv(N, 256),
                block_dim=256,
            )
            self._ctx.value().synchronize()
        # CPU path:
        else:
            # func() must take a device_type, so we must convert each argument
            # to device_type even for the CPU path.
            # _to_device_type() mutates a pointer, so we must allocate it first
            var dv = stack_allocation[1, V1.device_type]()
            # _to_device_type() takes a void pointer, so we need to cast
            v1._to_device_type(dv.bitcast[NoneType]())

            fn wrapper(i: Int) capturing -> None:
                func(i, dv[0])
            parallelize[wrapper](N)

    fn parallel_for[
        N: Int,
        V1: DevicePassable,
        V2: DevicePassable,
        func: fn(i: Int, v1: V1.device_type, v2: V2.device_type) -> None,
    ](mut self, v1: V1, v2: V2) raises:
        @parameter
        if Self.backend == "gpu":
            fn kernel(v1: V1.device_type, v2: V2.device_type):
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                if i < N:
                    func(i, v1, v2)

            self._ctx.value().enqueue_function[kernel, kernel](
                v1, v2,
                grid_dim=ceildiv(N, 256),
                block_dim=256,
            )
            self._ctx.value().synchronize()
        # CPU path:
        else:
            var dv1 = stack_allocation[1, V1.device_type]()
            var dv2 = stack_allocation[1, V2.device_type]()

            v1._to_device_type(dv1.bitcast[NoneType]())
            v2._to_device_type(dv2.bitcast[NoneType]())

            fn wrapper(i: Int) capturing -> None:
                func(i, dv1[0], dv2[0])
            parallelize[wrapper](N)

    fn parallel_for[
        N: Int,
        V1: DevicePassable,
        V2: DevicePassable,
        V3: DevicePassable,
        func: fn(i: Int, v1: V1.device_type, v2: V2.device_type, v3: V3.device_type) -> None,
    ](mut self, v1: V1, v2: V2, v3: V3) raises:
        @parameter
        if Self.backend == "gpu":
            fn kernel(v1: V1.device_type, v2: V2.device_type, v3: V3.device_type):
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                if i < N:
                    func(i, v1, v2, v3)

            self._ctx.value().enqueue_function[kernel, kernel](
                v1, v2, v3,
                grid_dim=ceildiv(N, 256),
                block_dim=256,
            )
            self._ctx.value().synchronize()
        # CPU path:
        else:
            var dv1 = stack_allocation[1, V1.device_type]()
            var dv2 = stack_allocation[1, V2.device_type]()
            var dv3 = stack_allocation[1, V3.device_type]()

            v1._to_device_type(dv1.bitcast[NoneType]())
            v2._to_device_type(dv2.bitcast[NoneType]())
            v3._to_device_type(dv3.bitcast[NoneType]())

            fn wrapper(i: Int) capturing -> None:
                func(i, dv1[0], dv2[0], dv3[0])
            parallelize[wrapper](N)


    fn parallel_reduce[
        N: Int,
        V1: DevicePassable,
        V2: DevicePassable,
        dtype: DType,
        func: fn(i: Int, v1: V1.device_type, v2: V2.device_type) -> Scalar[dtype],
    ](mut self, v1: V1, v2: V2) raises -> Scalar[dtype]:
        comptime num_threads = 256
        comptime num_blocks = ceildiv(N, num_threads)
        var res: Scalar[dtype] = 0
        @parameter
        if Self.backend == "gpu":
            partial = self._ctx.value().enqueue_create_buffer[dtype](num_blocks)

            fn kernel(
                v1: V1.device_type,
                v2: V2.device_type,
                partial: UnsafePointer[Scalar[dtype], MutAnyOrigin]
            ):
                var shared = stack_allocation[
                    num_threads,
                    Scalar[dtype],
                    address_space=AddressSpace.SHARED,
                ]()
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                var i_local = Int(thread_idx.x)
                if i < N:
                    shared[thread_idx.x] = func(i, v1, v2)
                barrier()

                var offset = num_threads // 2
                while offset > 0:
                    if i_local < offset:
                        shared[i_local] += shared[i_local + offset]
                    barrier()
                    offset >>= 1

                if i_local == 0:
                    partial[block_idx.x] = shared[0]

            self._ctx.value().enqueue_function[kernel, kernel](
                v1, v2, partial,
                grid_dim=num_blocks,
                block_dim=num_threads
            )
            self._ctx.value().synchronize()

            with partial.map_to_host() as h_partial:
                for i in range(num_blocks):
                    res += h_partial[i]
        # CPU path:
        # else:


        return res
