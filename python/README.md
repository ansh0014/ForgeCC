# ForgeCC — High-Performance Tensor Acceleration Engine

ForgeCC is a lightweight, high-performance tensor compiler and GPU execution engine for machine learning. It allows you to run GPU-accelerated tensor math and neural network operations directly on your NVIDIA GPU with **zero CUDA Toolkit or compiler installations**.

---

## 1. What is ForgeCC?

When running PyTorch or TensorFlow on a GPU, developers typically need to install multi-gigabyte CUDA Toolkits, configure `nvcc` compiler paths, and manage complex build environments.

ForgeCC communicates directly with your standard NVIDIA graphics card driver (`nvcuda.dll` on Windows, `libcuda.so` on Linux). It generates optimized GPU machine code on the fly and executes it immediately, delivering native GPU acceleration straight from Python.

---

## 2. Prerequisites & System Requirements

ForgeCC is designed for instant setup with minimal requirements:

- **Operating System:** Windows 10/11 (64-bit) or Linux (x86_64)
- **Python:** Version 3.8 or higher
- **NVIDIA GPU:** Any NVIDIA GPU with driver installed (GeForce, RTX, GTX, Tesla, Quadro)
- **CUDA Toolkit Installed:** **Not required** (ForgeCC uses your existing display driver)
- **C++ Compiler Installed:** **Not required** for Python package usage
- **Fallback:** If no NVIDIA GPU is detected, ForgeCC automatically runs on your CPU using multi-threaded SIMD routines.

---

## 3. How ForgeCC is Different & Why Use It

| Feature | Standard PyTorch | ForgeCC |
| :--- | :--- | :--- |
| **Setup Size** | Requires 2 GB – 5 GB download + CUDA SDK | Under 1 MB lightweight pip install |
| **CUDA Toolkit Required** | Yes (`nvcc`, `cudart`, large SDK) | **No** (Direct graphics driver communication) |
| **Kernel Execution** | Launches separate kernels with memory roundtrips | Fuses operations directly in GPU registers |
| **ONNX Runtime** | Requires `onnx` and `protobuf` libraries | Zero dependencies built-in parser |
| **PyTorch Acceleration** | Requires custom C++ toolchain / Triton | One-line decorator `@compile_torch_module` |
| **CPU Fallback** | Separate CPU wheels | Automatic built-in fallback |

### Key Benefits for Daily ML Work:

1. **Instant GPU Acceleration:** Start running GPU code on any machine with an NVIDIA card immediately after `pip install forgecc`.
2. **Reduced Memory & Higher Speed:** Fuses matrix multiplications and activations into a single GPU pass, keeping data in fast registers rather than writing back to VRAM.
3. **Seamless PyTorch Integration:** Speed up your existing PyTorch models with a single line of code.
4. **Zero-Dependency ONNX Inference:** Load and execute `.onnx` models without installing external runtime packages.

---

## 4. Installation

```bash
pip install forgecc
```

Verify installation:

```python
import forgecc
print("GPU Available:", forgecc.is_gpu_available())
```

---

## 5. Quickstart & Practical Examples

### 5.1 Basic Tensor Operations on GPU

```python
import forgecc
import numpy as np

a = forgecc.tensor([[1.0, 2.0], [3.0, 4.0]], device=forgecc.Device.GPU)
b = forgecc.tensor([[5.0, 6.0], [7.0, 8.0]], device=forgecc.Device.GPU)

c = a @ b
d = (c + 1.0).relu()

print("Result:\n", d.numpy())
```

---

### 5.2 Fast Operator Fusion (MatMul + Bias + ReLU + Residual)

Instead of launching 4 separate kernels, ForgeCC executes the entire sequence in a single GPU pass:

```python
import forgecc
import numpy as np

M, K, N = 512, 512, 512
a = np.random.randn(M, K).astype(np.float32)
b = np.random.randn(K, N).astype(np.float32)
bias = np.random.randn(N).astype(np.float32)
residual = np.random.randn(M, N).astype(np.float32)

output = forgecc.matmul_add_relu_residual(a, b, bias, residual)
print("Output shape:", output.shape)
```

---

### 5.3 Accelerating Existing PyTorch Models

Accelerate standard PyTorch `nn.Module` networks with zero changes to your model architecture:

```python
import torch
import torch.nn as nn
import forgecc

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = NeuralNet()
sample_input = torch.randn(1, 4)

compiled_model = forgecc.compile_torch_module(model, sample_input, target="gpu")
output = compiled_model(sample_input)

print("Output:", output)
```

---

### 5.4 Training Neural Networks with Autograd & Optimizers

Train models using ForgeCC's built-in automatic differentiation and optimizers (SGD, Adam, AdamW):

```python
import forgecc
import numpy as np

x = forgecc.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], requires_grad=False)
y_true = forgecc.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=False)

w1 = forgecc.tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
w2 = forgecc.tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)

optimizer = forgecc.optim.AdamW([w1, w2], lr=0.01)

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

### 5.5 Loading and Running ONNX Models Without Dependencies

```python
import forgecc

model = forgecc.compile_onnx("model.onnx", target="gpu")
input_data = forgecc.tensor([[1.0, 2.0, 3.0, 4.0]])

result = model(input_data)
print("Inference Output:\n", result.numpy())
```

---

### 5.6 C++ Native Universal API

Embed ForgeCC directly in your C++ applications:

```cpp
#include "forgecc_rt/gpu_runtime.hpp"
#include "forgecc_rt/tensor.hpp"
#include <iostream>
#include <vector>

int main() {
    int64_t M = 1024, K = 1024, N = 1024;
    
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C(M * N, 0.0f);

    if (forge_gpu_is_available()) {
        void* d_A = forge_gpu_malloc(M * K * sizeof(float));
        void* d_B = forge_gpu_malloc(K * N * sizeof(float));
        void* d_C = forge_gpu_malloc(M * N * sizeof(float));

        forge_gpu_memcpy_to_device(d_A, h_A.data(), M * K * sizeof(float));
        forge_gpu_memcpy_to_device(d_B, h_B.data(), K * N * sizeof(float));

        forge_gpu_matmul_relu_fused((const float*)d_A, (const float*)d_B, (float*)d_C, M, K, N);
        forge_gpu_sync();

        forge_gpu_memcpy_to_host(h_C.data(), d_C, M * N * sizeof(float));

        forge_gpu_free(d_A);
        forge_gpu_free(d_B);
        forge_gpu_free(d_C);

        std::cout << "Computed first element: " << h_C[0] << std::endl;
    }
    return 0;
}
```

---

## 6. GPU Performance Benchmark

Compare standard unfused PyTorch execution against ForgeCC fused kernel execution:

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

Monitor GPU load while running:

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv -l 1
```

---

## 7. License

ForgeCC is open-source software licensed under the MIT License.
