from std.testing import assert_equal, assert_true, TestSuite
from std.sys import has_accelerator

from mojito import *


# ===------------------------------------------------------------------=== #
# Generic test bodies, instantiated per target
# ===------------------------------------------------------------------=== #


def _test_create_and_fill[target: StaticString]() raises:
    comptime dtype = DType.float32
    var mj = Mojito[target]()
    var n = 100

    var a = mj.full[dtype](3.5, n)
    var m = mj.create_mirror(a)
    mj.deep_copy(m, a)
    mj.fence()
    for i in range(n):
        assert_equal(m[i], 3.5)

    assert_equal(a.size(), n)
    assert_equal(a.extent(0), n)

    var b3 = mj.full[dtype](-1.0, 2, 3, 4)
    assert_equal(b3.size(), 24)
    assert_equal(b3.extent(2), 4)
    var m3 = mj.create_mirror(b3)
    mj.deep_copy(m3, b3)
    mj.fence()
    for i in range(24):
        assert_equal(m3[i], -1.0)


def _test_mirror_semantics[target: StaticString]() raises:
    comptime dtype = DType.float32
    var mj = Mojito[target]()
    var a = mj.full[dtype](7.0, 10)
    var m = mj.create_mirror(a)

    comptime if target == "cpu":
        # Host-accessible source: the mirror aliases it.
        m[0] = 9.0
        assert_equal(a[0], 9.0)
    # deep_copy is a no-op on aliases and a real copy otherwise.
    mj.deep_copy(m, a)
    mj.fence()
    for i in range(1, 10):
        assert_equal(m[i], 7.0)


def _test_parallel_for_1d[target: StaticString]() raises:
    comptime dtype = DType.float32
    var mj = Mojito[target]()
    var n = 1000

    var x = mj.full[dtype](3.0, n)
    var y = mj.full[dtype](1.0, n)
    var xv = x.view()
    var yv = y.view()
    var alpha = Float32(2.0)

    def axpy(i: Int) {var alpha, var xv, var yv}:
        yv[i] = alpha * xv[i] + yv[i]

    mj.parallel_for(n, axpy)
    mj.fence()

    var m = mj.create_mirror(y)
    mj.deep_copy(m, y)
    mj.fence()
    for i in range(n):
        assert_equal(m[i], 7.0)


def _test_parallel_for_range_offset[target: StaticString]() raises:
    comptime dtype = DType.float32
    var mj = Mojito[target]()
    var n = 100

    var a = mj.full[dtype](0.0, n)
    var av = a.view()

    def body(i: Int) {var av}:
        av[i] = Float32(i)

    mj.parallel_for(RangePolicy(10, 20), body)
    mj.fence()

    var m = mj.create_mirror(a)
    mj.deep_copy(m, a)
    mj.fence()
    for i in range(n):
        if i >= 10 and i < 20:
            assert_equal(m[i], Float32(i))
        else:
            assert_equal(m[i], 0.0)


def _test_parallel_for_2d[target: StaticString]() raises:
    comptime dtype = DType.float32
    var mj = Mojito[target]()
    var nx = 37
    var ny = 53

    var a = mj.full[dtype](0.0, nx, ny)
    var av = a.view()

    def body(i: Int, j: Int) {var av}:
        av[i, j] = Float32(i * 1000 + j)

    mj.parallel_for(MDRangePolicy[2]({nx, ny}), body)
    mj.fence()

    var m = mj.create_mirror(a)
    mj.deep_copy(m, a)
    mj.fence()
    for i in range(nx):
        for j in range(ny):
            assert_equal(m[i, j], Float32(i * 1000 + j))


def _test_parallel_for_3d[target: StaticString]() raises:
    comptime dtype = DType.float32
    var mj = Mojito[target]()
    var nx = 8
    var ny = 9
    var nz = 10

    var a = mj.full[dtype](0.0, nx, ny, nz)
    var av = a.view()

    def body(i: Int, j: Int, k: Int) {var av}:
        av[i, j, k] = Float32((i * 100 + j) * 100 + k)

    mj.parallel_for(MDRangePolicy[3]({nx, ny, nz}), body)
    mj.fence()

    var m = mj.create_mirror(a)
    mj.deep_copy(m, a)
    mj.fence()
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                assert_equal(m[i, j, k], Float32((i * 100 + j) * 100 + k))


def _test_reduce_sum[target: StaticString]() raises:
    # float32: Apple GPUs have no fp64 support in Metal.
    comptime dtype = DType.float32
    var mj = Mojito[target]()
    var n = 10000

    var x = mj.full[dtype](3.0, n)
    var y = mj.full[dtype](2.0, n)
    var xv = x.view()
    var yv = y.view()

    def dot(i: Int) {var xv, var yv} -> Float32:
        return xv[i] * yv[i]

    var res = mj.parallel_reduce[Sum, dtype](n, dot)
    assert_equal(res, 6.0 * Float32(n))


def _test_reduce_minmax[target: StaticString]() raises:
    comptime dtype = DType.float32
    var mj = Mojito[target]()
    var n = 5000

    var a = mj.full[dtype](0.0, n)
    var av = a.view()

    def init(i: Int) {var av}:
        av[i] = Float32((i * 7919) % 10007)

    mj.parallel_for(n, init)
    mj.fence()

    def rd(i: Int) {var av} -> Float32:
        return av[i]

    var lo = mj.parallel_reduce[Min, dtype](n, rd)
    var hi = mj.parallel_reduce[Max, dtype](n, rd)

    # Reference on the host.
    var m = mj.create_mirror(a)
    mj.deep_copy(m, a)
    mj.fence()
    var ref_lo = Float32(1e30)
    var ref_hi = Float32(-1e30)
    for i in range(n):
        var v = m[i]
        if v < ref_lo:
            ref_lo = v
        if v > ref_hi:
            ref_hi = v
    assert_equal(lo, ref_lo)
    assert_equal(hi, ref_hi)


def _test_reduce_non_multiple[target: StaticString]() raises:
    comptime dtype = DType.float32
    var mj = Mojito[target]()

    # Sizes that are not multiples of the block size.
    for n in [1, 7, 255, 257, 1000, 4097]:
        var a = mj.full[dtype](1.0, n)
        var av = a.view()

        def one(i: Int) {var av} -> Float32:
            return av[i]

        var c = mj.parallel_reduce[Sum, dtype](n, one)
        assert_equal(c, Float32(n))


# ===------------------------------------------------------------------=== #
# CPU-only extras
# ===------------------------------------------------------------------=== #


def test_cpu_host_indexing() raises:
    comptime dtype = DType.float32
    var mj = Mojito["cpu"]()

    var a = mj.empty[dtype](4, 5)
    for i in range(4):
        for j in range(5):
            a[i, j] = Float32(i * 5 + j)
    for i in range(4):
        for j in range(5):
            assert_equal(a[i, j], Float32(i * 5 + j))

    var b = mj.empty[dtype](2, 3, 4)
    for i in range(24):
        b[i] = Float32(i)
    assert_equal(b[1, 2, 3], Float32(23))


# ===------------------------------------------------------------------=== #
# Per-target test entry points
# ===------------------------------------------------------------------=== #


def test_cpu_create_and_fill() raises:
    _test_create_and_fill["cpu"]()


def test_cpu_mirror_semantics() raises:
    _test_mirror_semantics["cpu"]()


def test_cpu_parallel_for_1d() raises:
    _test_parallel_for_1d["cpu"]()


def test_cpu_parallel_for_range_offset() raises:
    _test_parallel_for_range_offset["cpu"]()


def test_cpu_parallel_for_2d() raises:
    _test_parallel_for_2d["cpu"]()


def test_cpu_parallel_for_3d() raises:
    _test_parallel_for_3d["cpu"]()


def test_cpu_reduce_sum() raises:
    _test_reduce_sum["cpu"]()


def test_cpu_reduce_minmax() raises:
    _test_reduce_minmax["cpu"]()


def test_cpu_reduce_non_multiple() raises:
    _test_reduce_non_multiple["cpu"]()


def test_gpu_create_and_fill() raises:
    _test_create_and_fill["gpu"]()


def test_gpu_mirror_semantics() raises:
    _test_mirror_semantics["gpu"]()


def test_gpu_parallel_for_1d() raises:
    _test_parallel_for_1d["gpu"]()


def test_gpu_parallel_for_range_offset() raises:
    _test_parallel_for_range_offset["gpu"]()


def test_gpu_parallel_for_2d() raises:
    _test_parallel_for_2d["gpu"]()


def test_gpu_parallel_for_3d() raises:
    _test_parallel_for_3d["gpu"]()


def test_gpu_reduce_sum() raises:
    _test_reduce_sum["gpu"]()


def test_gpu_reduce_minmax() raises:
    _test_reduce_minmax["gpu"]()


def test_gpu_reduce_non_multiple() raises:
    _test_reduce_non_multiple["gpu"]()


def main():
    comptime if not has_accelerator():
        var suite = TestSuite(cli_args=List[StaticString]())
        suite.test[test_cpu_create_and_fill]()
        suite.test[test_cpu_mirror_semantics]()
        suite.test[test_cpu_host_indexing]()
        suite.test[test_cpu_parallel_for_1d]()
        suite.test[test_cpu_parallel_for_range_offset]()
        suite.test[test_cpu_parallel_for_2d]()
        suite.test[test_cpu_parallel_for_3d]()
        suite.test[test_cpu_reduce_sum]()
        suite.test[test_cpu_reduce_minmax]()
        suite.test[test_cpu_reduce_non_multiple]()
        try:
            suite^.run()
        except e:
            print("\nre-raised error:", e)
    else:
        try:
            TestSuite.discover_tests[__functions_in_module()]().run()
        except e:
            print("\nre-raised error:", e)
