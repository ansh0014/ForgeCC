from .core import (
    Device,
    is_gpu_available,
    sync,
    Tensor,
    tensor,
    matmul,
    relu,
    matmul_relu
)

__version__ = "0.1.3"
__all__ = [
    "Device",
    "is_gpu_available",
    "sync",
    "Tensor",
    "tensor",
    "matmul",
    "relu",
    "matmul_relu"
]
