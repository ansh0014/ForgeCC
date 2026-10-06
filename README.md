# ForgeCC: High-Performance AI and Tensor Compiler with GPU Acceleration

ForgeCC is a domain-specific compiler and universal acceleration runtime designed to compile and execute high-throughput tensor operations directly on NVIDIA GPUs and multi-threaded CPUs. Built on LLVM 22 and a zero-dependency dynamic CUDA engine.

## Key Highlights

- LLVM-Powered Code Generation: Emits optimized LLVM IR and compiles native machine object files (.o / .exe).
- Zero-SDK CUDA Acceleration: Features a dynamic CUDA Driver runtime with embedded PTX kernels. Accelerated GPU execution requires no CUDA Toolkit installation.
- Graph Optimization and Operator Fusion: Fuses compute-heavy operations (such as MatMul + ReLU) directly into GPU registers, eliminating intermediate global memory traffic.
- Universal C/C++ and Python Library: Framework-agnostic. Can be embedded directly into standalone C++ inference engines or imported into Python alongside PyTorch and NumPy.
- Automatic Device Fallback: Detects NVIDIA GPU availability at runtime and seamlessly falls back to CPU execution if no GPU is present.

## Compiler Architecture

```
                    ┌───────────────────────────────┐
                    │    Source / C++ / Python      │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   AST & Semantic Analysis     │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      ForgeCC Tensor IR        │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Optimization Pipeline     │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     LLVM CodeGen Backend      │
                    └───────────────┬───────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
    ┌───────────────────────────┐       ┌───────────────────────────┐
    │     CPU Runtime Engine    │       │    CUDA Driver Engine     │
    │  (Multi-Threaded Math)    │       │   (Embedded GPU PTX)      │
    └───────────────────────────┘       └───────────────────────────┘
```

## Quick Start

### Building ForgeCC

Prerequisites:
- Windows 10/11 with MSYS2 (ucrt64 environment)
- GCC / G++ (C++17 support)
- LLVM 22 Development Headers
- CMake 3.20+

Build Command:
```cmd
scripts\build.bat
```

Fast Incremental Rebuild:
```cmd
scripts\rebuild.bat
```

Run Verification Tests:
```cmd
scripts\run-tests.bat
```

## Usage Modes

### Mode 1: Python Package

```bash
cd python
pip install -e .
```

```python
import forgecc
import numpy as np

print("GPU Available:", forgecc.is_gpu_available())

A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)

C = forgecc.matmul_relu(A, B)
print("Result shape:", C.shape)
```

### Mode 2: Modern C++ Universal API

```cpp
#include "forgecc/forgecc.hpp"
#include <iostream>

int main() {
    forge::Device dev = forge::has_gpu() ? forge::Device::GPU : forge::Device::CPU;

    forge::Tensor input({1024, 1024}, dev);
    forge::Tensor weights({1024, 1024}, dev);

    input.from_host(host_input_ptr);
    weights.from_host(host_weights_ptr);

    forge::Tensor output = forge::matmul_relu(input, weights);
    forge::sync();

    std::vector<float> result(1024 * 1024);
    output.to_host(result.data());

    return 0;
}
```

### Mode 3: Standalone Compiler CLI

```cmd
build\tools\forgecc\forgecc.exe model.fcc --emit-forge-ir
build\tools\forgecc\forgecc.exe model.fcc --emit-ir
build\tools\forgecc\forgecc.exe model.fcc --emit-obj -o model_out
```

## License

ForgeCC is released under the MIT License.
