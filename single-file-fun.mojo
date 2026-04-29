from std.builtin.device_passable import DevicePassable

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys import has_accelerator

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

# Function bodies to test parallel_for (1, 2, 3 args)
def fill_body(
    i: Int,
    a: array_ref[DType.float32, 10],
) -> None:
    a[i] = Float32(i)

def parallel_for[
    Nx: Int,
    V1: DevicePassable,
    func: def(i: Int, v1: V1.device_type) thin -> None,
](ctx: DeviceContext, v1: V1) raises:
    def kernel(v1: V1.device_type):
        var i = Int(block_idx.x * block_dim.x + thread_idx.x)
        if i < Nx:
            func(i, v1)

    ctx.enqueue_function[kernel, kernel](
        v1,
        grid_dim = (ceildiv(Nx, TBSize)),
        block_dim = TBSize
        )
    ctx.synchronize()

def main() raises:

    ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](10)
    buf.enqueue_fill(1.0)
    a = array_ref[DType.float32, 10](buf.take_ptr())

    parallel_for[10, func=fill_body](ctx, a)