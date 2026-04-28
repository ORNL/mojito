from std.testing import assert_equal, TestSuite
from std.memory import UnsafePointer
from mojito import *

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys import has_accelerator


# Function bodies to test parallel_for (1, 2, 3 args)
def fill_body(
    i: Int,
    a: array_ref[DType.float32, 10],
) -> None:
    a[i] = Float32(i)


def test_gpu_parallel_for_1_arg() raises:
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


def main():
    comptime if not has_accelerator():
        var suite = TestSuite(cli_args=List[StaticString]())
        try:
            suite^.run()
        except e:
            print("\nre-raised error:", e)
    else:
        try:
            TestSuite.discover_tests[__functions_in_module()]().run()
        except e:
            print("\nre-raised error:", e)
