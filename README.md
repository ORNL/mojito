# mojito: simple performance portability in Mojo

[![ci-cpu](https://github.com/ORNL/mojito/actions/workflows/ci-cpu.yaml/badge.svg)](https://github.com/ORNL/mojito/actions/workflows/ci-cpu.yaml)
[![ci-gpu-NVIDIA](https://github.com/ORNL/mojito/actions/workflows/ci-gpu-NVIDIA.yaml/badge.svg)](https://github.com/ORNL/mojito/actions/workflows/ci-gpu-NVIDIA.yaml)
[![ci-gpu-AMD](https://github.com/ORNL/mojito/actions/workflows/ci-gpu-AMD.yaml/badge.svg)](https://github.com/ORNL/mojito/actions/workflows/ci-gpu-AMD.yaml)
[![ci-gpu-Apple](https://github.com/ORNL/mojito/actions/workflows/ci-gpu-Apple.yaml/badge.svg)](https://github.com/ORNL/mojito/actions/workflows/ci-gpu-Apple.yaml)


`mojito` is a [Mojo](https://www.modular.com/open-source/mojo) library to easily implement CPU/GPU performance portable `array`, `parallel_for` and `parallel_reduce` kernels. 

Mojo is a new programming language suporting performance portable low-level GPU kernel programming. `mojito` leverages Mojo by providing a high-level API to implement performance portable parallel CPU/GPU array memory and kernel launching by switching between `cpu` and the `gpu` [vendor backends supported by Mojo](https://docs.modular.com/max/packages/#gpu-compatibility).

## Getting started

1. Install [the Mojo language](https://docs.modular.com/mojo/manual/install/).
2. Clone the mojito repository and navigate to the project directory.
3. Run the tests to verify the installation:
   
   ```bash
   $ pixi run mojo test.mojo
   Running 15 tests for /home/wfg/workspace/mojito/test.mojo 
    PASS [ 0.006 ] test_cpu_arrays
    PASS [ 0.001 ] test_cpu_init
    PASS [ 323.629 ] test_gpu_arrays
    PASS [ 0.157 ] test_gpu_kernel
    PASS [ 0.019 ] test_3D_gpu_arrays
    PASS [ 0.003 ] test_cpu_copy_functions
    PASS [ 0.023 ] test_gpu_copy_to_host
    PASS [ 0.026 ] test_gpu_copy_to_device
    PASS [ 0.444 ] test_cpu_parallel_for_1_arg
    PASS [ 0.095 ] test_gpu_parallel_for_1_arg
    PASS [ 0.318 ] test_cpu_parallel_for_2_args
    PASS [ 0.079 ] test_gpu_parallel_for_2_args
    PASS [ 0.293 ] test_cpu_parallel_for_3_args
    PASS [ 0.091 ] test_gpu_parallel_for_3_args
    PASS [ 0.102 ] test_gpu_parallel_reduce_2_args
    --------
   Summary [ 325.291 ] 15 tests run: 15 passed , 0 failed , 0 skipped 
   ```

Code API example:

```mojo
from mojito import *

def main() raises:
    var mj = Mojito()   # "gpu" if an accelerator is present, else "cpu"
    var n = 100         # array extents are runtime values

    var x = mj.full[DType.float32](3.0, n)
    var y = mj.full[DType.float32](1.0, n)
    var xv = x.view()
    var yv = y.view()
    var alpha = Float32(2.0)

    # Kernel bodies are closures that capture views BY VALUE (`var`).
    def axpy(i: Int) {var alpha, var xv, var yv}:
        yv[i] = alpha * xv[i] + yv[i]

    mj.parallel_for(n, axpy)   # asynchronous on GPU; fence() awaits
    mj.fence()

    def dot(i: Int) {var xv, var yv} -> Float32:
        return xv[i] * yv[i]

    # Reducers are monoids: Sum, Prod, Min, Max (or your own ReduceOp).
    var d = mj.parallel_reduce[Sum, DType.float32](n, dot)

    # Mirrors may alias when data is already host-accessible;
    # deep_copy always copies (no-op on the same allocation).
    var m = mj.create_mirror(y)
    mj.deep_copy(m, y)
    mj.fence()
    for i in range(n):
        print(m[i])  # should print 7.0
```

## Known issues

- Current version pinned to `Mojo==1.0.0` and `MAX==26.5.0` (the `max` conda package provides the `max` Mojo package, where `DeviceContext` and other GPU host APIs live as of Mojo 1.0)
- GPU launches are asynchronous; CPU launches currently block on return.
- Apple GPUs have no fp64 support in Metal; use `DType.float32` there.
- Apple M1/M3 GPU support requires running `xcodebuild -downloadComponent MetalToolchain`, see [issue](https://github.com/modular/modular/issues/6466). 

## Project status

The project is in an early exploratory development stage and follows closely nightly changes in the Mojo language. The API is not stable and may change without deprecation. We welcome contributions and feedback to help shape the direction of the project. Please reach out to us if you are interested in contributing or have any questions by opening an issue.

## Sponsor

The work is funded by the Advanced Scientific Computing Research (ASCR) program within the U.S. Department of Energy's Office of Science. S4PST and MAGMA/Fairbanks projects and Oak Ridge National Laboratory internship programs.

# Contributors

- [Tatiana Melnichenko](https://github.com/tdehoff), University of Tennessee, Knoxville
- [William F Godoy](https://github.com/williamfgc), Oak Ridge National Laboratory