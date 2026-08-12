from mojito import Mojito, array_ref
from std.sys import argv, has_accelerator
from std.sys.info import size_of
from std.time import monotonic
from std.utils.numerics import max_finite, min_finite

comptime dtype = DType.float64
comptime SIZE = 1 << 25
# comptime SIZE = 10240
comptime num_iter = 1000

comptime initA: Scalar[dtype] = 0.1
comptime initB: Scalar[dtype] = 0.2
comptime initC: Scalar[dtype] = 0.0
comptime startScalar: Scalar[dtype] = 0.4

def copy_body(
    i: Int,
    a: array_ref[dtype, SIZE],
    c: array_ref[dtype, SIZE],
) -> None:
    c[i] = a[i]

def mul_body(
    i: Int,
    b: array_ref[dtype, SIZE],
    c: array_ref[dtype, SIZE],
) -> None:
    b[i] = startScalar * c[i]

def add_body(
    i: Int,
    a: array_ref[dtype, SIZE],
    b: array_ref[dtype, SIZE],
    c: array_ref[dtype, SIZE],
) -> None:
    c[i] = a[i] + b[i]

def triad_body(
    i: Int,
    a: array_ref[dtype, SIZE],
    b: array_ref[dtype, SIZE],
    c: array_ref[dtype, SIZE],
) -> None:
    a[i] = b[i] + startScalar * c[i]

def dot_body(
    i: Int,
    a: array_ref[dtype, SIZE],
    b: array_ref[dtype, SIZE],
) -> Scalar[dtype]:
    return a[i] * b[i]


def run[backend: String]() raises:
    var csv_output = False
    var args = argv()
    var i = 0
    while i < len(args):
        var arg = args[i]
        if arg == "--csv":
            csv_output = True
        i += 1

    var mj = Mojito[backend]()

    var a = mj.fill[dtype, SIZE](initA)
    var b = mj.fill[dtype, SIZE](initB)
    var c = mj.fill[dtype, SIZE](initC)

    var timings = List[Float64](length=5 * num_iter, fill=0.0)

    for i in range(num_iter):
        var start = monotonic()
        mj.parallel_for[SIZE, func=copy_body](a, c)
        var end = monotonic()
        timings[0 * num_iter + i] = Float64(end - start)

        start = monotonic()
        mj.parallel_for[SIZE, func=mul_body](b, c)
        end = monotonic()
        timings[1 * num_iter + i] = Float64(end - start)
        # print("MUL", i, "timig:", Float64(end-start) * 10e-9, "sec")

        start = monotonic()
        mj.parallel_for[SIZE, func=add_body](a, b, c)
        end = monotonic()
        timings[2 * num_iter + i] = Float64(end - start)
        # print("ADD", i, "timig:", Float64(end-start) * 10e-9, "sec")

        start = monotonic()
        mj.parallel_for[SIZE, func=triad_body](a, b, c)
        end = monotonic()
        timings[3 * num_iter + i] = Float64(end - start)
        # print("TRIAD", i, "timig:", Float64(end-start) * 10e-9, "sec")

        start = monotonic()
        var res = mj.parallel_reduce[SIZE, dtype=dtype, func=dot_body](a, b)
        end = monotonic()
        timings[4 * num_iter + i] = Float64(end - start)

    var bytes_per_elem = size_of[Scalar[dtype]]()
    # Copy: 2N, Mul: 2N, Add: 3N, Triad: 3N, Dot: 2N
    var kernel_data: List[Int] = [
        2 * SIZE * bytes_per_elem,
        2 * SIZE * bytes_per_elem,
        3 * SIZE * bytes_per_elem,
        3 * SIZE * bytes_per_elem,
        2 * SIZE * bytes_per_elem,
    ]
    var kernel_names = ["Copy", "Mul", "Add", "Triad", "Dot"]

    if csv_output:
        print("backend,GPU,precision,vec_size,routine,BW_GBs")
        for k in range(5):
            for it in range(1, num_iter):
                print("Mojo,", backend, ",", dtype, ",", SIZE, ",", kernel_names[k], ",", Float64(kernel_data[k]) / timings[k * num_iter + it])
    else:
        print("Backend:", backend)
        print("Array size:", Float64(SIZE * bytes_per_elem) * 1e-6, "MB")
        print("Total size:", Float64(3 * SIZE * bytes_per_elem) * 1e-6, "MB")
        for k in range(5):
            var min_t: Float64 = max_finite[dtype]()
            var max_t: Float64 = min_finite[dtype]()
            var mean_t: Float64 = 0

            # Ignore warmup timing
            for it in range(1, num_iter):
                var t = timings[k * num_iter + it]
                if t < min_t:
                    min_t = t
                if t > max_t:
                    max_t = t
                mean_t += t

            mean_t /= (num_iter - 1)
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
