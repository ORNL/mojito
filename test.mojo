from testing import assert_equal, TestSuite
from memory import UnsafePointer
from mojito import *

# 0.26.1.0.dev2026011405 (1796fa39)

from gpu import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from math import ceildiv

fn init_kernel_gpu[
    dtype: DType
](
    Nx: Int,
    inout_array: UnsafePointer[Scalar[dtype], MutAnyOrigin]
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < UInt(Nx):
        inout_array[i] = Scalar[dtype](i)


fn init_kernel_cpu[
    dtype: DType
](
    Nx: Int,
    inout_array: UnsafePointer[Scalar[dtype], MutAnyOrigin]
):
    for i in range(Nx):
        inout_array[i] = Scalar[dtype](i)


def test_cpu_arrays():
    comptime backend = "cpu"
    comptime Nx = 5
    comptime dtype = DType.float32

    mj = Mojito[backend]()

    mj_arr = mj.empty[dtype, Nx]()
    mj_zeros = mj.zeros[dtype, Nx]()
    mj_ones = mj.ones[dtype, Nx]()
    mj_fill = mj.fill[dtype, Nx](-1.0)

    for i in range(Nx):
        mj_arr[i] = Scalar[dtype](i)

    for i in range(Nx):
        assert_equal(mj_arr[i], Scalar[dtype](i))
        assert_equal(mj_zeros[i], 0)
        assert_equal(mj_ones[i], 1)
        assert_equal(mj_fill[i], -1)


def test_cpu_init():
    comptime backend = "cpu"
    comptime Nx = 5
    comptime dtype = DType.float32

    mj = Mojito[backend]()
    mj_arr = mj.zeros[dtype, Nx]()

    init_kernel_cpu[dtype](Nx, mj_arr._data)

    for i in range(Nx):
        assert_equal(mj_arr[i], Scalar[dtype](i))


def test_gpu_arrays():
    comptime backend = "gpu"
    comptime Nx = 5
    comptime dtype = DType.float32

    mj = Mojito[backend]()

    mj_zeros = mj.zeros[dtype, Nx]()
    mj_ones = mj.ones[dtype, Nx]()
    mj_fill = mj.fill[dtype, Nx](-1.0)

    mj_zeros.to_host()
    mj_ones.to_host()
    mj_fill.to_host()

    mj.sync()

    for i in range(Nx):
        assert_equal(mj_zeros[i], 0)
        assert_equal(mj_ones[i], 1)
        assert_equal(mj_fill[i], -1.0)


def test_gpu_kernel():
    comptime backend = "gpu"
    comptime Nx = 5
    comptime dtype = DType.float32

    mj = Mojito[backend]()
    mj_arr = mj.empty[dtype, Nx]()

    ctx = mj.get_ctx()
    comptime kernel = init_kernel_gpu[dtype]
    ctx.enqueue_function[kernel, kernel](
        Nx,
        mj_arr._data,
        grid_dim=ceildiv(Nx, 256),
        block_dim=256
    )
    mj.sync()

    mj_arr.to_host()
    mj.sync()

    for i in range(Nx):
        assert_equal(mj_arr[i], Scalar[dtype](i))


def test_3D_gpu_arrays():
    comptime backend = "gpu"
    comptime Nx = 2
    comptime Ny = 3
    comptime Nz = 4
    comptime dtype = DType.float32

    mj = Mojito[backend]()

    mj_arr = mj.empty[dtype, Nx, Ny, Nz]()

    mj_arr.to_host()
    mj.sync()

    for i in range (Nx * Ny * Nz):
        mj_arr[i] = Scalar[dtype](i)

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                assert_equal(mj_arr[i, j, k], Scalar[dtype](i * Ny * Nz + j * Nz + k))


def test_cpu_copy_functions():
    comptime backend = "cpu"
    comptime Nx = 5
    comptime dtype = DType.float32

    mj = Mojito[backend]()
    mj_arr = mj.fill[dtype, Nx](3.0)

    host_copy = mj.copy_to_host(mj_arr)
    dev_copy = mj.copy_to_device(mj_arr)

    # Both are shallow (non-owning); values match the original
    for i in range(Nx):
        assert_equal(host_copy[i], 3.0)
        assert_equal(dev_copy[i], 3.0)

    # Original is intact
    for i in range(Nx):
        assert_equal(mj_arr[i], 3.0)


def test_gpu_copy_to_host():
    comptime backend = "gpu"
    comptime Nx = 5
    comptime dtype = DType.float32

    mj = Mojito[backend]()
    dev_arr = mj.fill[dtype, Nx](7.0)  # buffer starts on device

    # Deep copy: new host buffer, dev_arr stays on device
    host_copy = mj.copy_to_host(dev_arr)
    mj.sync()

    for i in range(Nx):
        assert_equal(host_copy[i], 7.0)

    # host_copy owns its buffer and knows it is on host
    assert_equal(host_copy._owned, True)
    assert_equal(host_copy._on_host, True)

    # original is still on device
    assert_equal(dev_arr._on_host, False)


def test_gpu_copy_to_device():
    comptime backend = "gpu"
    comptime Nx = 5
    comptime dtype = DType.float32

    mj = Mojito[backend]()

    # Build a host-pinned source array with known values
    host_arr = mj.fill[dtype, Nx](0.0)
    host_arr.to_host()
    mj.sync()
    for i in range(Nx):
        host_arr[i] = Scalar[dtype](i)

    # Deep copy to device
    dev_copy = mj.copy_to_device(host_arr)

    # Move back to verify values
    dev_copy.to_host()
    mj.sync()
    for i in range(Nx):
        assert_equal(dev_copy[i], Scalar[dtype](i))

    # Original host array is unchanged
    for i in range(Nx):
        assert_equal(host_arr[i], Scalar[dtype](i))

# Function bodies to test parallel_for (1, 2, 3 args)
fn fill_body(
    i: Int,
    a: array_ref[DType.float32, 10],
) -> None:
    a[i] = Float32(i)

fn copy_body(
    i: Int,
    a: array_ref[DType.float32, 10],
    b: array_ref[DType.float32, 10],
) -> None:
    a[i] = b[i]

fn axpy_body(
    i: Int,
    alpha: Float32,
    x: array_ref[DType.float32, 10],
    y: array_ref[DType.float32, 10],
) -> None:
    y[i] = alpha * x[i] + y[i]


def test_cpu_parallel_for_1_arg():
    comptime backend = "cpu"
    comptime N = 10
    comptime dtype = DType.float32

    mj = Mojito[backend]()
    a = mj.zeros[dtype, N]()

    mj.parallel_for[N, func=fill_body](a)

    for i in range(N):
        assert_equal(a[i], Scalar[dtype](i))


def test_gpu_parallel_for_1_arg():
    comptime backend = "gpu"
    comptime N = 10
    comptime dtype = DType.float32

    mj = Mojito[backend]()
    a = mj.zeros[dtype, N]()
    mj.parallel_for[N, func=fill_body](a)

    a.to_host()
    mj.sync()

    for i in range(N):
        assert_equal(a[i], Scalar[dtype](i))


def test_cpu_parallel_for_2_args():
    comptime backend = "cpu"
    comptime N = 10
    comptime dtype = DType.float32

    mj = Mojito[backend]()
    a = mj.zeros[dtype, N]()
    b = mj.fill[dtype, N](7.0)

    mj.parallel_for[N, func=copy_body](a, b)

    for i in range(N):
        assert_equal(a[i], b[i])


def test_gpu_parallel_for_2_args():
    comptime backend = "gpu"
    comptime N = 10
    comptime dtype = DType.float32

    mj = Mojito[backend]()
    a = mj.zeros[dtype, N]()
    b = mj.fill[dtype, N](7.0)

    mj.parallel_for[N, func=copy_body](a, b)

    a.to_host()
    b.to_host()
    mj.sync()

    for i in range(N):
        assert_equal(a[i], b[i])


def test_cpu_parallel_for_3_args():
    comptime backend = "cpu"
    comptime N = 10
    comptime dtype = DType.float32

    mj = Mojito[backend]()
    alpha = Float32(2.0)
    x = mj.fill[dtype, N](3.0)
    y = mj.fill[dtype, N](1.0)

    mj.parallel_for[N, func=axpy_body](alpha, x, y)

    # y[i] = 2.0 * 3.0 + 1.0 = 7.0
    for i in range(N):
        assert_equal(y[i], 7.0)


def test_gpu_parallel_for_3_args():
    comptime backend = "gpu"
    comptime N = 10
    comptime dtype = DType.float32

    mj = Mojito[backend]()
    alpha = Float32(2.0)
    x = mj.fill[dtype, N](3.0)
    y = mj.fill[dtype, N](1.0)

    mj.parallel_for[N, func=axpy_body](alpha, x, y)

    y.to_host()
    mj.sync()

    # y[i] = 2.0 * 3.0 + 1.0 = 7.0
    for i in range(N):
        assert_equal(y[i], 7.0)


fn body(
    i: Int,
    a: array_ref[DType.float32, 10],
    b: array_ref[DType.float32, 10]
) -> Float32:
    return (a[i] * b[i])

def test_gpu_parallel_reduce_2_args():
    comptime backend = "gpu"
    comptime N = 10
    comptime dtype = DType.float32

    mj = Mojito[backend]()
    x = mj.fill[dtype, N](3.0)
    y = mj.fill[dtype, N](2.0)

    var res = mj.parallel_reduce[N, dtype=dtype, func=body](x, y)

    # 10 * 3.0 * 2.0 = 60.0
    assert_equal(res, Float32(60.0))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
