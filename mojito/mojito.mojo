from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.builtin.device_passable import DevicePassable

from std.memory import stack_allocation
from std.collections import Optional
from std.algorithm import parallelize, reduction
from std.math import ceildiv

comptime TBSize = 512

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

    def __init__(
        out self,
        data: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
    ):
        self._data = data

    # Followed syntax from https://github.com/modular/modular/issues/6145
    def _to_device_type[
        mut_origin: Origin[mut=True]
    ](
        self,
        target: UnsafePointer[NoneType, mut_origin]
    ):
        target.bitcast[Self]().init_pointee_copy(self)

    @staticmethod
    def get_type_name() -> String:
        return "MojitoArrayView"

    @staticmethod
    def get_device_type_name() -> String:
        return "MojitoArrayView"

    # Getters and setters for GPU kernel array manipulation
    def __getitem__(self, x: Int) -> Scalar[Self.dtype]:
        return self._data[x]

    def __setitem__(self, x: Int, value: Scalar[Self.dtype]):
        self._data[x] = value



struct array[
    backend: String,
    dtype: DType,
    Nx: Int,
    Ny: Int = 1,
    Nz: Int = 1,
](DevicePassable, ImplicitlyCopyable):
    comptime _N = Self.Nx * Self.Ny * Self.Nz
    var _ctx : Optional[DeviceContext]
    var _data: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var _on_host: Bool
    var _owned: Bool

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

  
    # 1D indexing
    def __getitem__(ref self, x: Int) raises -> Scalar[Self.dtype]:
        return self._data[x]

    # 1D setitem
    def __setitem__(mut self, x: Int, value: Scalar[Self.dtype]) raises:
        self._data[x] = value

    # Move device buffer to host for GPU backend, does nothing in other cases
    def to_host(mut self) raises:
        if self._ctx and not self._on_host:
            var h_buff = self._ctx.value().enqueue_create_host_buffer[Self.dtype](Self._N)
            h_buff.enqueue_copy_from(self._data)
            self._data = h_buff.take_ptr()
            self._on_host = True

    # DevicePassable requirements
    comptime device_type = array_ref[Self.dtype, Self.Nx, Self.Ny, Self.Nz]

    def _view(self) -> array_ref[Self.dtype, Self.Nx, Self.Ny, Self.Nz]:
        return array_ref[Self.dtype, Self.Nx, Self.Ny, Self.Nz](self._data)

    def _to_device_type[
        mut_origin: Origin[mut=True]
    ](
        self,
        target: UnsafePointer[NoneType, mut_origin]
    ):
        target.bitcast[array_ref[Self.dtype, Self.Nx, Self.Ny, Self.Nz]]()
            .init_pointee_copy(self._view())

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

    def zeros[
        type: DType,
        Nx: Int,
        Ny: Int = 1,
        Nz: Int = 1
    ](mut self) raises -> array[Self.backend, type, Nx, Ny, Nz]:
        return array[Self.backend, type, Nx, Ny, Nz](self._ctx.value(), Scalar[type](0))
        
    # Synchronize DeviceContext for GPU backend
    def sync(mut self) raises:
        if self._ctx:
            self._ctx.value().synchronize()


    # parallel_for overloads for 1, 2, and 3 arguments
    def parallel_for[
        Nx: Int,
        V1: DevicePassable,
        func: def(i: Int, v1: V1.device_type) thin -> None,
    ](mut self, v1: V1) raises:
        comptime if Self.backend == "gpu":
            def kernel(v1: V1.device_type):
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                if i < Nx:
                    func(i, v1)

            self._ctx.value().enqueue_function[kernel, kernel](
                v1,
                grid_dim = (ceildiv(Nx, TBSize)),
                block_dim = TBSize
            )
            self._ctx.value().synchronize()


  