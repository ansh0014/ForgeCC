"""
ForgeCC: LLVM & CUDA Accelerated AI / Tensor Compiler Engine
"""

from .core import (
    Device,
    is_gpu_available,
    matmul_relu
)

__version__ = "0.1.0"
__all__ = ["Device", "is_gpu_available", "matmul_relu"]
