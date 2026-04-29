from std.gpu.host import DeviceContext
from std.sys import has_accelerator, has_apple_gpu_accelerator, num_physical_cores

def main() raises:
    if not has_accelerator():
        print("\nDoes not detect accelerator")
    else:
        print("Detects accelerator")
        if has_apple_gpu_accelerator():
            print("\nDetects Apple GPU")
            ctx = DeviceContext()
            print("\nPhysical cores: ", num_physical_cores())
        else:
            print("\nDetects non-Apple GPU")
        