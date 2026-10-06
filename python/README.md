# ForgeCC: High-Performance AI & Tensor Compiler with GPU Acceleration

**ForgeCC** is a domain-specific compiler and universal acceleration runtime designed to compile and execute high-throughput tensor operations directly on NVIDIA GPUs and multi-threaded CPUs. Built on **LLVM 22** and a **zero-dependency dynamic CUDA engine**.

## Installation
```bash
pip install forgecc
```

## Quickstart
```python
import forgecc
import numpy as np

print("GPU Available:", forgecc.is_gpu_available())

A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)

C = forgecc.matmul_relu(A, B)
print("Computed result shape:", C.shape)
```
