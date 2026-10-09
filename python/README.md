# ForgeCC — High-Performance Tensor Acceleration Engine

ForgeCC is a lightweight, high-performance tensor acceleration runtime and machine learning compiler designed for NVIDIA GPUs and multi-threaded CPUs.

ForgeCC enables instant GPU execution for deep learning operations without requiring the NVIDIA CUDA Toolkit or external C++ compilers. It communicates directly with standard NVIDIA display drivers and JIT-compiles optimized PTX machine code on the fly.

---

## Key Features

- **Zero-SDK GPU Acceleration:** Run GPU-accelerated tensor math directly using your standard NVIDIA graphics card driver (`nvcuda.dll` on Windows, `libcuda.so` on Linux). No CUDA Toolkit installation required.
- **Register-Level Operator Fusion:** Fuses matrix multiplications, bias additions, activations, and residual connections into single-pass GPU kernels, eliminating intermediate memory roundtrips.
- **Seamless PyTorch Acceleration:** Compile and accelerate standard PyTorch `nn.Module` networks using the `@compile_torch_module` decorator or `torch.compile` backend.
- **Zero-Dependency ONNX Inference:** Load and execute `.onnx` models with an internal binary Protocol Buffer parser without installing third-party runtime dependencies.
- **Built-in Autograd & Optimizers:** Train models end-to-end with reverse-mode automatic differentiation and native in-place optimizers (SGD, Adam, AdamW).
- **Automatic CPU Fallback:** Automatically detects GPU presence and routes execution to multi-threaded SIMD CPU routines if an NVIDIA GPU is not available.

---

## Feature Comparison Matrix

| Feature | Standard PyTorch / TensorFlow | TorchInductor / Triton | TensorRT | ForgeCC Compiler Engine |
| :--- | :--- | :--- | :--- | :--- |
| **CUDA SDK Dependency** | Requires CUDA Toolkit (3 GB – 10 GB) | Requires CUDA Toolkit + C++ Compiler | Requires CUDA Toolkit + TensorRT SDK | **Zero SDK requirement**: communicates directly with standard graphics driver |
| **Compiler Toolchain** | Heavy native wheel builds | Requires MSVC/GCC and Triton runtime | Proprietary engine | **Zero external compiler**: generates clean PTX assembly and JIT-loads via driver API |
| **Execution Model** | Eager kernel dispatch | JIT Triton compilation | Graph capture engine | **Dynamic SSA IR JIT + Register-Fused PTX Kernels** |
| **ONNX Parsing** | Requires `onnx` and `protobuf` libraries | PyTorch importer only | TensorRT parser | **Zero-dependency binary Protobuf decoder** built directly into runtime |
| **Memory Management** | Framework caching allocator | Framework caching allocator | Static engine workspace | **Bucketed Power-of-Two GPU Memory Pool** with zero driver reallocation |
| **Fallback Mechanism** | Requires separate CPU builds | CPU compilation limited | No CPU fallback | **Seamless automatic fallback** to multi-threaded C++ SIMD engine |
| **Universal API** | Python / C++ LibTorch | Python only | C++ / Python | **Unified C++ universal interface and Python package** |

---

## Prerequisites

- **Operating System:** Windows 10 / 11 (64-bit) or Linux (x86_64)
- **Python:** Version 3.8 or higher
- **GPU (Optional):** Any NVIDIA GPU with display drivers installed (GeForce, RTX, GTX, Quadro, Tesla)
- **CUDA Toolkit:** Not required
- **C++ Compiler:** Not required

---

## Installation

```bash
pip install forgecc
```

Verify GPU availability:

```python
import forgecc

print("ForgeCC Version:", forgecc.__version__)
print("GPU Available:", forgecc.is_gpu_available())
```

---

## Performance Benchmarks

Benchmarking standard PyTorch eager execution against ForgeCC fused PTX kernel execution on an NVIDIA GPU:

| Operation | Matrix Size | Standard PyTorch (Unfused) | ForgeCC (Fused PTX) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **MatMul + ReLU** | 1024 x 1024 | 0.42 ms | 0.12 ms | **3.50x faster** |
| **MatMul + ReLU** | 2048 x 2048 | 1.82 ms | 0.48 ms | **3.79x faster** |
| **MatMul + Bias + ReLU + Residual** | 2048 x 2048 | 2.41 ms | 0.59 ms | **4.08x faster** |
| **Elementwise Add + ReLU** | 4096 x 4096 | 1.15 ms | 0.28 ms | **4.10x faster** |

### Benchmark Reproduction Script

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

torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
for _ in range(ITERATIONS):
    _ = torch.relu(torch.matmul(A_torch, B_torch))
torch.cuda.synchronize()
pytorch_time = (time.perf_counter() - start) * 1000 / ITERATIONS
pytorch_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

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

---

## Quickstart & Practical Examples

### 1. Basic Tensor Operations on GPU

```python
import forgecc

a = forgecc.tensor([[1.0, 2.0], [3.0, 4.0]], device=forgecc.Device.GPU)
b = forgecc.tensor([[5.0, 6.0], [7.0, 8.0]], device=forgecc.Device.GPU)

c = a @ b
d = (c + 1.0).relu()

print("Result:\n", d.numpy())
```

---

### 2. High-Performance Operator Fusion

Execute Matrix Multiplication, Bias Add, ReLU activation, and Residual Add in a single GPU kernel launch:

```python
import forgecc
import numpy as np

M, K, N = 512, 512, 512
a = np.random.randn(M, K).astype(np.float32)
b = np.random.randn(K, N).astype(np.float32)
bias = np.random.randn(N).astype(np.float32)
residual = np.random.randn(M, N).astype(np.float32)

output = forgecc.matmul_add_relu_residual(a, b, bias, residual)
print("Fused Output Shape:", output.shape)
```

---

### 3. Accelerating PyTorch Modules

Accelerate existing PyTorch models with zero architectural changes:

```python
import torch
import torch.nn as nn
import forgecc

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = MLP()
sample_input = torch.randn(1, 4)

compiled_model = forgecc.compile_torch_module(model, sample_input, target="gpu")
output = compiled_model(sample_input)

print("Compiled Output:", output)
```

---

### 4. Neural Network Training with Autograd & AdamW

```python
import forgecc
import numpy as np

x = forgecc.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], requires_grad=False)
y_true = forgecc.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=False)

w1 = forgecc.tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
w2 = forgecc.tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)

optimizer = forgecc.optim.AdamW([w1, w2], lr=0.01, weight_decay=0.01)

for epoch in range(50):
    optimizer.zero_grad()
    
    hidden = (x @ w1).relu()
    y_pred = (hidden @ w2).sigmoid()
    
    loss = forgecc.mse_loss(y_pred, y_true)
    loss.backward()
    
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {float(loss.numpy().item()):.6f}")
```

---

### 5. Standalone ONNX Model Inference

Run ONNX models directly without installing `onnx` or `onnxruntime`:

```python
import forgecc

model = forgecc.compile_onnx("model.onnx", target="gpu")
input_data = forgecc.tensor([[1.0, 2.0, 3.0, 4.0]])

result = model(input_data)
print("Inference Output:\n", result.numpy())
```

---

## License

ForgeCC is distributed under the [MIT License](https://github.com/ansh0014/ForgeCC/blob/main/LICENSE).
