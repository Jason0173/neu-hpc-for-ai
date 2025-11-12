# deepseek_moe/build.py
from torch.utils.cpp_extension import load
import os

_this_dir = os.path.dirname(__file__)
sources = [os.path.join(_this_dir, "csrc", "moe_bindings.cpp"),
           os.path.join(_this_dir, "csrc", "moe_kernels.cu")]
moe_cuda = load(name="moe_cuda", sources=sources, extra_cuda_cflags=["-O3"], verbose=False)
