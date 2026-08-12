from std.sys import argv, has_accelerator
from std.sys.info import size_of, CompilationTarget
from std.time import monotonic
from std.utils.numerics import max_finite, min_finite

from mojito import *

# Apple GPUs have no fp64 support in Metal; BabelStream reference uses fp64.
comptime dtype = DType.float64 if not CompilationTarget.is_apple_silicon() else DType.float32
comptime num_iter = 1000

comptime initA: Scalar[dtype] = 0.1
comptime initB: Scalar[dtype] = 0.2
comptime initC: Scalar[dtype] = 0.0
comptime startScalar: Scalar[dtype] = 0.4


def run[target: StaticString]() raises:
    var csv_output = False
    var args = argv()
    var i = 0
    while i < len(args):
        var arg = args[i]
        if arg == "--csv":
            csv_output = True
        i += 1

    var mj = Mojito[target]()
    var size = 1 << 25

    var a = mj.full[dtype](initA, size)
    var b = mj.full[dtype](initB, size)
    var c = mj.full[dtype](initC, size)
    var av = a.view()
    var bv = b.view()
    var cv = c.view()
    mj.fence()

    def copy_body(i: Int) {var av, var cv}:
        cv[i] = av[i]

    def mul_body(i: Int) {var bv, var cv}:
        bv[i] = startScalar * cv[i]

    def add_body(i: Int) {var av, var bv, var cv}:
        cv[i] = av[i] + bv[i]

    def triad_body(i: Int) {var av, var bv, var cv}:
        av[i] = bv[i] + startScalar * cv[i]

    def dot_body(i: Int) {var av, var bv} -> Scalar[dtype]:
        return av[i] * bv[i]

    var timings = List[Float64](length=5 * num_iter, fill=0.0)

    for it in range(num_iter):
        var start = monotonic()
        mj.parallel_for(size, copy_body)
        mj.fence()
        var end = monotonic()
        timings[0 * num_iter + it] = Float64(end - start)

        start = monotonic()
        mj.parallel_for(size, mul_body)
        mj.fence()
        end = monotonic()
        timings[1 * num_iter + it] = Float64(end - start)

        start = monotonic()
        mj.parallel_for(size, add_body)
        mj.fence()
        end = monotonic()
        timings[2 * num_iter + it] = Float64(end - start)

        start = monotonic()
        mj.parallel_for(size, triad_body)
        mj.fence()
        end = monotonic()
        timings[3 * num_iter + it] = Float64(end - start)

        start = monotonic()
        var res = mj.parallel_reduce[Sum, dtype](size, dot_body)
        end = monotonic()
        timings[4 * num_iter + it] = Float64(end - start)
        _ = res

    var bytes_per_elem = size_of[Scalar[dtype]]()
    # Copy: 2N, Mul: 2N, Add: 3N, Triad: 3N, Dot: 2N
    var kernel_data: List[Int] = [
        2 * size * bytes_per_elem,
        2 * size * bytes_per_elem,
        3 * size * bytes_per_elem,
        3 * size * bytes_per_elem,
        2 * size * bytes_per_elem,
    ]
    var kernel_names = ["Copy", "Mul", "Add", "Triad", "Dot"]

    if csv_output:
        print("backend,GPU,precision,vec_size,routine,BW_GBs")
        for k in range(5):
            for it in range(1, num_iter):
                print(
                    "Mojo,",
                    target,
                    ",",
                    dtype,
                    ",",
                    size,
                    ",",
                    kernel_names[k],
                    ",",
                    Float64(kernel_data[k]) / timings[k * num_iter + it],
                )
    else:
        print("Backend:", target)
        print("Array size:", Float64(size * bytes_per_elem) * 1e-6, "MB")
        print("Total size:", Float64(3 * size * bytes_per_elem) * 1e-6, "MB")
        for k in range(5):
            var min_t = Float64(max_finite[DType.float64]())
            var max_t = Float64(min_finite[DType.float64]())
            var mean_t: Float64 = 0

            # Ignore warmup timing
            for it in range(1, num_iter):
                var t = timings[k * num_iter + it]
                if t < min_t:
                    min_t = t
                if t > max_t:
                    max_t = t
                mean_t += t

            mean_t /= num_iter - 1
            print(kernel_names[k], ":")
            print("   Min (sec):", min_t * 1e-9)
            print("   Max (sec):", max_t * 1e-9)
            print("   Avg (sec):", mean_t * 1e-9)
            print("   Bandwidth (GB/s):", Float64(kernel_data[k]) / min_t)


def main() raises:
    comptime if has_accelerator():
        run["gpu"]()
        run["cpu"]()
    else:
        run["cpu"]()
