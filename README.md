# mojito

`mojito` is a [Mojo](https://www.modular.com/open-source/mojo) library to easily implement CPU/GPU performance portable `array`, `parallel_for` and `parallel_reduce` kernels. 

Mojo is a new programming language suporting performance portable low-level GPU kernel programming. `mojito` leverages Mojo by providing a high-level API to implement performance portable parallel CPU/GPU array and kernels by switching between backends.

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

comptime N = 100

def axpy_kernel(
    i: Int,
    alpha: Float32,
    x: array_ref[DType.float32, N],
    y: array_ref[DType.float32, N],
) -> None:
    y[i] = alpha * x[i] + y[i]


def main();

    comptime backend = "gpu" # or "cpu"
    mj = Mojito[backend]()
    alpha = Float32(2.0)
    x = mj.fill[dtype, N](3.0)
    y = mj.ones[dtype, N]()

    mj.parallel_for[N, func=axpy_kernel](alpha, x, y)

    y.to_host() # move data back to host if running on GPU
    mj.sync()

    # y[i] = 2.0 * 3.0 + 1.0 = 7.0
    for i in range(N):
        print(y[i]) # should print 7.0
```

## Known issues

- Currently works with mojo = "==0.26.3.0.dev2026042205"
- Apple M1/M3 GPU support fixed in Mojo nightly. 
- To do: CI to be added with Mojo v1.0 release.

## Project status

The project is in an early exploratory development stage and follows closely nightly changes in the Mojo language. The API is not stable and may change without deprecation. We welcome contributions and feedback to help shape the direction of the project. Please reach out to us if you are interested in contributing or have any questions by opening an issue.

## Sponsor

The work is funded by the Advanced Scientific Computing Research (ASCR) program within the U.S. Department of Energy's Office of Science. S4PST and MAGMA/Fairbanks projects and Oak Ridge National Laboratory internship programs.

# Contributors

- [Tatiana Melnichenko](https://github.com/tdehoff), University of Tennessee Knoxville
- [William F Godoy](https://github.com/williamfgc), Oak Ridge National Laboratory