from mojito import Mojito, array_ref
from std.sys import has_accelerator
from std.sys.info import size_of
from std.time import monotonic

comptime L = 512
comptime num_iter = 1000
comptime dtype = DType.float32

comptime c = Float32(0.5)
comptime h: Scalar[dtype] = 1.0 / (L - 1)

def init_body(
    ix: Int, iy: Int, iz: Int,
    u: array_ref[dtype, L, L, L],
) -> None:
    var x = Scalar[dtype](ix) * h
    var y = Scalar[dtype](iy) * h
    var z = Scalar[dtype](iz) * h
    var Lx = Scalar[dtype](L) * h
    var Ly = Scalar[dtype](L) * h
    var Lz = Scalar[dtype](L) * h
    u[ix, iy, iz] = c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz)

comptime invh2: Scalar[dtype] = 1 / h / h
comptime invhxyz2: Scalar[dtype] = -2.0 * 3.0 * invh2

def laplacian_body(
    ix: Int, iy: Int, iz: Int,
    f: array_ref[dtype, L, L, L],
    u: array_ref[dtype, L, L, L],
) -> None:
    if ix > 0 and ix < L - 1 and
       iy > 0 and iy < L - 1 and
       iz > 0 and iz < L - 1:
        f[ix, iy, iz] = u[ix, iy, iz] * invhxyz2
            + (u[ix - 1, iy, iz] + u[ix + 1, iy, iz]) * invh2
            + (u[ix, iy - 1, iz] + u[ix, iy + 1, iz]) * invh2
            + (u[ix, iy, iz - 1] + u[ix, iy, iz + 1]) * invh2


def run[backend: String]() raises:
    mj = Mojito[backend]()

    u = mj.empty[dtype, L, L, L]()
    f = mj.empty[dtype, L, L, L]()

    mj.parallel_for[L, L, L, func=init_body](u)

    var bytes_per_elem = size_of[Scalar[dtype]]()
    var fetch_size = (L * L * L - 8 - 4 * (L - 2) - 4 * (L - 2) - 4 * (L - 2)) * bytes_per_elem
    var write_size = ((L - 2) * (L - 2) * (L - 2)) * bytes_per_elem
    var datasize = fetch_size + write_size

    # Warmup
    mj.parallel_for[L, L, L, func=laplacian_body](f, u)

    var total_ns: UInt = 0
    for _ in range(num_iter):
        var start = monotonic()
        mj.parallel_for[L, L, L, func=laplacian_body](f, u)
        var end = monotonic()
        var elapsed = end - start
        # var bw_gbs = Float64(datasize) / Float64(elapsed)
        total_ns += elapsed

    print("Backend:", backend)
    print("L:", L)
    print("Average kernel time:", Float64(total_ns) / 1e6 / num_iter, "ms")
    print("Effective memory bandwidth:", Float64(datasize) * num_iter / Float64(total_ns), "GB/s")


def main() raises:
    comptime if has_accelerator():
        run["gpu"]()
        run["cpu"]()
    else:
        run["cpu"]()
