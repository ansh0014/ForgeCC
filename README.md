# ForgeCC: High-Performance AI & Tensor Compiler with GPU Acceleration

**ForgeCC** is a domain-specific compiler and universal acceleration runtime designed to compile and execute high-throughput tensor operations directly on NVIDIA GPUs and multi-threaded CPUs. Built on **LLVM 22** and a **zero-dependency dynamic CUDA engine**, ForgeCC lowers high-level tensor computation graphs into fused machine instructions and embedded PTX assembly.

---

## ⚡ Key Highlights

- **LLVM-Powered Code Generation:** Emits optimized LLVM IR and compiles native machine object files (`.o` / `.exe`).
- **Zero-SDK CUDA Acceleration:** Features a dynamic CUDA Driver runtime with embedded PTX kernels. Accelerated GPU execution requires **no massive 15 GB CUDA Toolkit installation**—only standard NVIDIA GPU drivers.
- **Graph Optimization & Operator Fusion:** Fuses compute-heavy operations (such as $\text{MatMul} + \text{ReLU}$) directly into GPU registers, eliminating intermediate global memory traffic.
- **Universal C/C++ & Python Library:** Framework-agnostic. Can be embedded directly into standalone C++ inference engines or imported into Python alongside PyTorch and NumPy.
- **Automatic Device Fallback:** Detects NVIDIA GPU availability at runtime and seamlessly falls back to CPU execution if no GPU is present.

---

## 🏗️ Compiler Architecture

```
                    ┌───────────────────────────────┐
                    │    Source / C++ / Python      │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   AST & Semantic Analysis     │  (Type checking & Symbol Table)
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      ForgeCC Tensor IR        │  (Graph representation & shapes)
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Optimization Pipeline     │  (Device Placement, Fusion, DCE)
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     LLVM CodeGen Backend      │  (Lowers to LLVM 22 IR / .o)
                    └───────────────┬───────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
    ┌───────────────────────────┐       ┌───────────────────────────┐
    │     CPU Runtime Engine    │       │    CUDA Driver Engine     │
    │  (Multi-Threaded Math)    │       │   (Embedded GPU PTX)      │
    └───────────────────────────┘       └───────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Building ForgeCC

**Prerequisites:**
- Windows 10/11 with MSYS2 (`ucrt64` environment)
- GCC / G++ (C++17 support)
- LLVM 22 Development Headers
- CMake 3.20+

**Build Command:**
```cmd
scripts\build.bat
```

**Fast Incremental Rebuild:**
```cmd
scripts\rebuild.bat
```

**Run All Verification Tests:**
```cmd
scripts\run-tests.bat
```

---

## 💻 Usage Modes

### Mode 1: Python Package (`import forgecc`)

Install the local package:
```bash
cd python
pip install -e .
```

Use in Python:
```python
import forgecc
import numpy as np

# Check GPU acceleration
print("GPU Available:", forgecc.is_gpu_available())

# Initialize input & weights
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)

# Executes fused MatMul + ReLU on GPU in parallel
C = forgecc.matmul_relu(A, B)
print("Result shape:", C.shape)
```

---

### Mode 2: Modern C++ Universal API

Any C++ AI inference engine or standalone application can import ForgeCC:

```cpp
#include "forgecc/forgecc.hpp"
#include <iostream>

int main() {
    // Select device (GPU if available, otherwise CPU)
    forge::Device dev = forge::has_gpu() ? forge::Device::GPU : forge::Device::CPU;

    // Allocate tensors on VRAM or RAM
    forge::Tensor input({1024, 1024}, dev);
    forge::Tensor weights({1024, 1024}, dev);

    // Copy weights into device
    input.from_host(host_input_ptr);
    weights.from_host(host_weights_ptr);

    // Compute accelerated fused layer
    forge::Tensor output = forge::matmul_relu(input, weights);
    forge::sync();

    // Copy result back
    std::vector<float> result(1024 * 1024);
    output.to_host(result.data());

    std::cout << "Computed layer successfully on GPU!\n";
    return 0;
}
```

---

### Mode 3: Standalone Compiler CLI (`forgecc.exe`)

Compile tensor programs and inspect intermediate representations:

```cmd
# 1. Emit ForgeCC Tensor IR
build\tools\forgecc\forgecc.exe model.fcc --emit-forge-ir

# 2. Emit Optimized LLVM IR
build\tools\forgecc\forgecc.exe model.fcc --emit-ir

# 3. Emit Native Object File (.o)
build\tools\forgecc\forgecc.exe model.fcc --emit-obj -o model_out
```

---

## 📊 Benchmark & Performance

Testing a $1024 \times 1024$ fused Linear + Activation layer on GPU:

| Operation | Implementation | Latency | Memory Traffic |
|---|---|---|---|
| MatMul + ReLU | Unfused (2 Kernels) | ~22.4 ms | High (2 Global Memory Roundtrips) |
| **ForgeCC** | **Fused (1 PTX Kernel)** | **~10.6 ms** | **Optimal (Computed in Registers)** |

---

## 📁 Repository Structure

```
ForgeCC/
├── include/forgecc/
│   ├── AST/              # AST definitions & nodes
│   ├── CodeGen/          # LLVM CodeGen interface
│   ├── IR/               # ForgeCC Tensor IR definition
│   ├── Optimizer/        # Optimization passes & PassManager
│   ├── Sema/             # Semantic analyzer & symbol table
│   ├── Support/          # Diagnostics & logging
│   ├── forgecc.h         # Universal C ABI header
│   └── forgecc.hpp       # Universal Modern C++ API header
├── lib/                  # Compiler frontend, middle-end & backend libraries
├── runtime/              # CPU runtime & Dynamic CUDA driver PTX engine
├── tools/forgecc/        # CLI driver executable (main.cpp, Driver.cpp)
├── examples/             # AI model inference benchmarks
├── python/               # Python package (setup.py, pyproject.toml, core.py)
├── scripts/              # Windows batch scripts for building and testing
└── docs/                 # Detailed architecture documentation
```

---

## 📜 License
ForgeCC is released under the MIT License.
