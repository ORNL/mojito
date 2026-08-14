from std.sys import has_accelerator
from std.sys.info import size_of
from std.time import monotonic

from mojito import *

comptime num_iter = 1000
comptime dtype = DType.float32

comptime c = Float32(0.5)


def run[target: StaticString]() raises:
    var mj = Mojito[target]()
    var L = 512

    var u = mj.empty[dtype](L, L, L)
    var f = mj.empty[dtype](L, L, L)
    var uv = u.view()
    var fv = f.view()

    var h: Scalar[dtype] = 1.0 / Scalar[dtype](L - 1)
    var invh2: Scalar[dtype] = 1 / h / h
    var invhxyz2: Scalar[dtype] = -2.0 * 3.0 * invh2

    def init_body(ix: Int, iy: Int, iz: Int) {var uv, var h, var L}:
        var x = Scalar[dtype](ix) * h
        var y = Scalar[dtype](iy) * h
        var z = Scalar[dtype](iz) * h
        var edge = Scalar[dtype](L) * h
        uv[ix, iy, iz] = (
            c * x * (x - edge) + c * y * (y - edge) + c * z * (z - edge)
        )

    def laplacian_body(
        ix: Int, iy: Int, iz: Int
    ) {var fv, var uv, var invh2, var invhxyz2, var L}:
        if (
            ix > 0
            and ix < L - 1
            and iy > 0
            and iy < L - 1
            and iz > 0
            and iz < L - 1
        ):
            fv[ix, iy, iz] = (
                uv[ix, iy, iz] * invhxyz2
                + (uv[ix - 1, iy, iz] + uv[ix + 1, iy, iz]) * invh2
                + (uv[ix, iy - 1, iz] + uv[ix, iy + 1, iz]) * invh2
                + (uv[ix, iy, iz - 1] + uv[ix, iy, iz + 1]) * invh2
            )

    var policy = MDRangePolicy[3]({L, L, L})
    mj.parallel_for(policy, init_body)
    mj.fence()

    var bytes_per_elem = size_of[Scalar[dtype]]()
    var fetch_size = (
        L * L * L - 8 - 4 * (L - 2) - 4 * (L - 2) - 4 * (L - 2)
    ) * bytes_per_elem
    var write_size = ((L - 2) * (L - 2) * (L - 2)) * bytes_per_elem
    var datasize = fetch_size + write_size

    # Warmup
    mj.parallel_for(policy, laplacian_body)
    mj.fence()

    var total_ns: Int = 0
    for _ in range(num_iter):
        var start = monotonic()
        mj.parallel_for(policy, laplacian_body)
        mj.fence()
        var end = monotonic()
        total_ns += end - start

    print("Backend:", target)
    print("L:", L)
    print("Average kernel time:", Float64(total_ns) / 1e6 / num_iter, "ms")
    print(
        "Effective memory bandwidth:",
        Float64(datasize) * num_iter / Float64(total_ns),
        "GB/s",
    )


def main() raises:
    comptime if has_accelerator():
        run["gpu"]()
        run["cpu"]()
    else:
        run["cpu"]()
