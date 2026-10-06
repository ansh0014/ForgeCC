# ForgeCC: High-Performance AI and Tensor Acceleration Engine

ForgeCC is a domain-specific compiler and acceleration runtime designed to compile and execute high-throughput neural network operations directly on NVIDIA GPUs and multi-threaded CPUs.

## Key Highlights

- **Zero-SDK Dynamic CUDA Execution:** Accelerated GPU execution requires no CUDA Toolkit installation. ForgeCC dynamically communicates with standard NVIDIA GPU drivers (nvcuda.dll).
- **Do Users Need to Install LLVM?**
  - **For Python / Pip users:** No. The runtime engine is precompiled and works out of the box with `pip install forgecc`.
  - **For C++ Developers:** LLVM 22 is only required if building the native compiler from source.
- **Operator Fusion:** Fuses Matrix Multiplications and Activations (such as MatMul + ReLU) directly inside GPU registers, eliminating global memory roundtrips.
- **Automatic Device Fallback:** Seamlessly detects GPU availability and falls back to multi-threaded CPU math if no NVIDIA GPU is present.

## Installation

```bash
pip install forgecc
```

## Quickstart Example

```python
import forgecc
import numpy as np

# 1. Check GPU acceleration status
print("GPU Available:", forgecc.is_gpu_available())

# 2. Create tensors (1024x1024 layer weights and input)
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)

# 3. Run accelerated fused MatMul + ReLU on GPU
C = forgecc.matmul_relu(A, B)

print("Computed Output Shape:", C.shape)
```

## How to Check GPU Usage (Before vs. After ForgeCC)

### 1. Benchmark Script

Run this script to benchmark Standard PyTorch (Unfused) vs. ForgeCC (Fused GPU Kernel):

```python
import time
import torch
import forgecc
import numpy as np

M, K, N = 2048, 2048, 2048
ITERATIONS = 100

A_np = np.random.randn(M, K).astype(np.float32)
B_np = np.random.randn(K, N).astype(np.float32)

A_torch = torch.from_numpy(A_np).cuda()
B_torch = torch.from_numpy(B_np).cuda()

# 1. BEFORE (Standard PyTorch GPU - 2 Separate Kernels)
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
for _ in range(ITERATIONS):
    _ = torch.relu(torch.matmul(A_torch, B_torch))
torch.cuda.synchronize()
pytorch_time = (time.perf_counter() - start) * 1000 / ITERATIONS
pytorch_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

# 2. AFTER (ForgeCC - 1 Fused GPU Register Kernel)
start = time.perf_counter()
for _ in range(ITERATIONS):
    _ = forgecc.matmul_relu(A_np, B_np)
forgecc_time = (time.perf_counter() - start) * 1000 / ITERATIONS

print("=" * 55)
print(f"BENCHMARK: Matrix Size ({M}x{K}) x ({K}x{N})")
print("=" * 55)
print(f"Standard PyTorch (Unfused): {pytorch_time:.3f} ms | Peak VRAM: {pytorch_mem:.2f} MB")
print(f"ForgeCC (Fused PTX Kernel): {forgecc_time:.3f} ms")
print(f"Speedup: {pytorch_time / forgecc_time:.2f}x faster")
print("=" * 55)
```

### 2. Live Hardware Monitoring

Open a second terminal to watch live GPU VRAM and compute utilization:

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv -l 1
```

## License

ForgeCC is distributed under the MIT License.
