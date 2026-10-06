# ForgeCC Technical Manual & Architecture Documentation

## 1. Executive Summary

ForgeCC is a domain-specific compiler and heterogeneous execution runtime for machine learning operations. It bridges high-level tensor computation graphs with hardware execution by combining LLVM backend lowering with a zero-overhead CUDA driver runtime.

---

## 2. Compiler Pipeline Specifications

### 2.1 Frontend & Semantic Analysis
- **AST Nodes (`ASTNodes.hpp`):** Represents declarations (`TensorDeclNode`), matrix operations (`MatMulNode`), activations (`ReLUNode`), and program containers (`ProgramNode`).
- **Semantic Analysis (`Sema.hpp`):** Performs symbol table registration, type resolution, operand validation, and automatic output tensor symbol registration.

### 2.2 Intermediate Representation (ForgeCC IR)
- **TensorIR (`TensorIR.hpp`):** Represents operations as directed graph nodes (`Operation`) operating on strongly-typed tensor values (`Value`).
- **Shapes and Data Types:** Multi-dimensional tensor shapes ($M \times K$, $K \times N$) with explicit data type specifications (`f32`, `i32`).

### 2.3 Optimization Passes (`Passes.hpp`)
1. **Device Placement Pass:** Analyzes operation intensity and targets tensor compute to `DeviceKind::GPU` or `DeviceKind::CPU`.
2. **Operator Fusion Pass:** Identifies producer-consumer chains (e.g. $\text{MatMul} \rightarrow \text{ReLU}$) and marks them for combined execution in GPU registers.
3. **Dead Operation Elimination Pass:** Removes unused intermediate tensors and disconnected ops.

### 2.4 Code Generation & LLVM Lowering (`CodeGen.cpp`)
- Lowers ForgeCC IR directly into LLVM 22 IR structures.
- Generates dynamic function calls to `forge_gpu_malloc`, `forge_gpu_matmul`, `forge_gpu_relu`, `forge_gpu_sync`, and `forge_gpu_free`.
- Target machine emission generates native `.o` object files with target triple detection.

---

## 3. GPU Acceleration Architecture

### 3.1 Zero-SDK Dynamic CUDA Loader (`gpu_runtime.cpp`)
Instead of linking against static CUDA libraries requiring the 15 GB NVIDIA CUDA Toolkit, ForgeCC dynamically queries `nvcuda.dll` on Windows (`libcuda.so` on Linux) at runtime:
- `cuInit`
- `cuDeviceGet`
- `cuCtxCreate`
- `cuMemAlloc` / `cuMemFree`
- `cuMemcpyHtoD` / `cuMemcpyDtoH`
- `cuModuleLoadData`
- `cuLaunchKernel`
- `cuCtxSynchronize`

### 3.2 Embedded PTX Kernel Execution
Optimized PTX assembly is embedded directly in the runtime binary:
- **`gpu_matmul_kernel`:** Matrix multiplication mapped across a 2D grid of thread blocks ($16 \times 16$).
- **`gpu_relu_kernel`:** Parallel elementwise ReLU activation.
- **`gpu_matmul_relu_kernel`:** Fused single-pass kernel that performs GEMM and applies activation in registers before global memory store.

---

## 4. API Reference

### 4.1 C ABI (`forgecc.h`)
```c
ForgeTensor* forge_tensor_create(const int64_t* shape, int ndim, ForgeDevice dev);
void         forge_tensor_free(ForgeTensor* t);
void         forge_tensor_copy_from_host(ForgeTensor* dst, const float* src_host);
void         forge_tensor_copy_to_host(const ForgeTensor* src, float* dst_host);
void         forge_op_matmul_relu(const ForgeTensor* A, const ForgeTensor* B, ForgeTensor* C);
bool         forge_has_gpu_support(void);
void         forge_sync(void);
```

### 4.2 C++ Modern Interface (`forgecc.hpp`)
```cpp
namespace forge {
    class Tensor;
    Tensor matmul(const Tensor& A, const Tensor& B);
    Tensor relu(const Tensor& in);
    Tensor matmul_relu(const Tensor& A, const Tensor& B);
    bool   has_gpu();
    void   sync();
}
```

### 4.3 Python Interface (`forgecc`)
```python
import forgecc
import numpy as np

forgecc.is_gpu_available() -> bool
forgecc.matmul_relu(A: np.ndarray, B: np.ndarray) -> np.ndarray
```

---

## 5. Build and Test Verification

### Scripts
- `scripts/build.bat`: Configures CMake with MSYS2 MinGW & LLVM 22 and builds all targets.
- `scripts/rebuild.bat`: Runs incremental build.
- `scripts/clean-build.bat`: Cleans build directory and reconfigures from scratch.
- `scripts/run-tests.bat`: Runs IR dumps, LLVM CodeGen, object file emission, and standard compilation modes.
