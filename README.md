# mojito

Branch to keep track of issue introduced with ==0.26.3.0.dev2026042305.
**Goal: reproduce outside mojito and report to Mojo team**

$ pixi run mojo test.mojo 

Works with 0.26.3.0.dev2026042205:

```
Running 1 tests for /home/wfg/tmp/mojito/test.mojo 
    PASS [ 455.861 ] test_gpu_parallel_for_1_arg
--------
Summary [ 455.861 ] 1 tests run: 1 passed , 0 failed , 0 skipped 
```

Fails with 0.26.3.0.dev2026042305:

```
$ pixi run mojo test.mojo 
Included from /home/wfg/tmp/mojito/mojito/__init__.mojo:1:
Included from /home/wfg/tmp/mojito/mojito/__init__.mojo:1:
/home/wfg/tmp/mojito/mojito/mojito.mojo:157:30: error: no matching method in call to 'enqueue_function'
            self._ctx.value().enqueue_function[kernel, kernel](
            ~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~
/home/wfg/tmp/mojito/test.mojo:1:1: note: candidate not viable: cannot use a dynamic value in a parameter list
from std.testing import assert_equal, TestSuite
^
/home/wfg/tmp/mojito/test.mojo:1:1: note: candidate not viable: cannot use a dynamic value in a parameter list
from std.testing import assert_equal, TestSuite
^
/home/wfg/tmp/mojito/test.mojo:1:1: note: candidate not viable: value passed to 'func' cannot be converted from 'def[__mlir_type.`!kgen.closure<@"mojito::mojito::Mojito::parallel_for[::Int,::DevicePassable,def(i: ::Int, v1: $1|2.device_type) -> None](mojito::mojito::Mojito[$0]&,$2)", "kernel" nonescaping>`, {}]' to 'func_type', argument type 'def[__mlir_type.`!kgen.closure<@"mojito::mojito::Mojito::parallel_for[::Int,::DevicePassable,def(i: ::Int, v1: $1|2.device_type) -> None](mojito::mojito::Mojito[$0]&,$2)", "kernel" nonescaping>`, {}]' does not conform to trait 'TrivialRegisterPassable'
from std.testing import assert_equal, TestSuite
^
/home/wfg/tmp/mojito/test.mojo:1:1: note: candidate not viable: value passed to 'func' cannot be converted from 'def[__mlir_type.`!kgen.closure<@"mojito::mojito::Mojito::parallel_for[::Int,::DevicePassable,def(i: ::Int, v1: $1|2.device_type) -> None](mojito::mojito::Mojito[$0]&,$2)", "kernel" nonescaping>`, {}]' to 'func_type', argument type 'def[__mlir_type.`!kgen.closure<@"mojito::mojito::Mojito::parallel_for[::Int,::DevicePassable,def(i: ::Int, v1: $1|2.device_type) -> None](mojito::mojito::Mojito[$0]&,$2)", "kernel" nonescaping>`, {}]' does not conform to trait 'TrivialRegisterPassable'
from std.testing import assert_equal, TestSuite
^
/home/wfg/tmp/mojito/test.mojo:1:1: note: candidate not viable: cannot use a dynamic value in a parameter list
from std.testing import assert_equal, TestSuite
^
/home/wfg/tmp/mojito/.pixi/envs/default/bin/mojo: error: failed to parse the provided Mojo source module
```
