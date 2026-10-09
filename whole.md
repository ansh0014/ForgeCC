# ForgeCC — Architecture, Internals, and Technical Specification

## 1. Executive Summary

ForgeCC is a framework-independent machine learning compiler and high-performance execution engine designed to execute neural network computation graphs on CPUs and NVIDIA GPUs.

Unlike conventional machine learning frameworks that depend on gigabyte-scale CUDA Toolkit installations, ForgeCC operates with zero external CUDA SDK dependencies by interfacing directly with the NVIDIA display driver (`nvcuda.dll` on Windows, `libcuda.so` on Linux) through dynamic runtime symbol loading.

The system incorporates an end-to-end compilation pipeline encompassing:
1. Multiple frontend entry points (Python API, zero-dependency binary ONNX importer, and PyTorch FX graph capture).
2. A Static Single Assignment (SSA) Tensor Intermediate Representation (IR).
3. An extensible optimization pass manager (Constant Folding, Algebraic Simplification, Common Subexpression Elimination, Dead Code Elimination, and Operator Fusion).
4. A PTX assembly code generator generating raw virtual instruction set architecture code targeting `sm_50` and above.
5. A bucketed power-of-two GPU memory pool eliminating driver allocation overhead.
6. A dynamic hardware-aware auto-tuner.
7. A reverse-mode automatic differentiation tape and native optimizer engine (SGD, Adam, AdamW).

---

## 2. System Architecture

```text
+-----------------------------------------------------------------------------+
|                                FRONTENDS                                    |
|  +---------------------+  +----------------------+  +--------------------+  |
|  |   ForgeCC Python    |  |  ONNX Binary Parser  |  |     PyTorch FX     |  |
|  |     Tensor API      |  |  (Zero Dependencies) |  |   Graph Capture    |  |
|  +----------+----------+  +----------+-----------+  +---------+----------+  |
+-------------|------------------------|------------------------|-------------+
              |                        |                        |
              +------------------------v------------------------+
                                       |
+--------------------------------------v--------------------------------------+
|                           INTERMEDIATE REPRESENTATION                       |
|  +-----------------------------------------------------------------------+  |
|  | SSA Graph: Graph, Node, Value (Typed Shapes, Constant Storage)        |  |
|  +-----------------------------------+-----------------------------------+  |
+--------------------------------------|--------------------------------------+
                                       |
+--------------------------------------v--------------------------------------+
|                           OPTIMIZATION PIPELINE                             |
|  +-----------------------------------------------------------------------+  |
|  | PassManager:                                                          |  |
|  |   1. Constant Folding Pass                                            |  |
|  |   2. Algebraic Simplification Pass                                    |  |
|  |   3. Common Subexpression Elimination Pass (CSE)                      |  |
|  |   4. Advanced Operator Fusion Pass (Epilogue + Residual Fusion)       |  |
|  |   5. Dead Code Elimination Pass (DCE)                                 |  |
|  +-----------------------------------+-----------------------------------+  |
+--------------------------------------|--------------------------------------+
                                       |
+--------------------------------------v--------------------------------------+
|                           BACKENDS & CODEGEN                                |
|  +-----------------------------------+-----------------------------------+  |
|  | CUDACodeGen (PTX sm_50+)          | CPU C++ Engine                    |  |
|  | Dynamic Driver JIT Launch         | Multithreaded C++ Runtime         |  |
|  +-----------------------------------+-----------------------------------+  |
+--------------------------------------|--------------------------------------+
                                       |
+--------------------------------------v--------------------------------------+
|                           EXECUTION RUNTIMES                                |
|  +-----------------------------------+-----------------------------------+  |
|  | GPU Runtime (nvcuda.dll Driver)   | CPU Runtime (libforgecc_rt.dll)   |  |
|  | Bucketed GPUMemoryPool            | Native CPU GEMM & Activations     |  |
|  | Hardware-Aware Auto-Tuning        | Native Reverse Autograd & Optim   |  |
|  +-----------------------------------+-----------------------------------+  |
+-----------------------------------------------------------------------------+
```

---

## 3. Core Modules & Codebase Structure

The repository is organized into a modular structure separating frontend Python interfaces, IR compilers, and low-level native runtime libraries.

```text
ForgeCC/
├── python/
│   └── forgecc/
│       ├── __init__.py           # Unified package exports and public API
│       ├── core.py               # Tensor, Device, GPUMemoryPool, ForgeRuntime, Driver Bindings
│       ├── ir.py                 # SSA IR Graph, Node, Value, Passes, CUDACodeGen, Tracer, Compiler
│       ├── onnx_frontend.py      # Zero-dependency binary Protobuf reader and ONNX graph importer
│       ├── adapters.py           # PyTorch FX GraphModule translator and module compiler
│       ├── autograd.py           # Reverse-mode differentiation tape and loss functions
│       ├── optim.py              # SGD, Adam, and AdamW optimizer implementations
│       └── libforgecc_rt.dll     # Compiled native runtime shared library (Windows)
├── runtime/
│   ├── include/
│   │   └── forgecc_rt/
│   │       ├── device.hpp        # Device enumeration definitions
│   │       ├── gpu_runtime.hpp   # C-linkage GPU driver interface declarations
│   │       └── tensor.hpp        # Low-level native tensor interfaces
│   └── src/
│       ├── cpu_runtime.cpp       # Native C++ CPU arithmetic, GEMM, activations, backward, and optimizers
│       ├── gpu_runtime.cpp       # Embedded PTX kernels, nvcuda dynamic loader, and GPU memory routines
│       └── universal_api.cpp     # Unified C runtime dispatch layer
├── working.md                    # Roadmap milestone tracking dashboard
└── CMakeLists.txt                # Build configuration for native C++ shared library
```

---

## 4. Technical Breakdown of All 10 Phases

### Phase 1: Tensor Operations & Memory Engine
* **Multidimensional Tensor Representation:** `forgecc.Tensor` manages dense multidimensional float32 arrays across CPU and GPU devices.
* **Device Management:** `Device.CPU` and `Device.GPU` enumerations with automatic hardware fallback when an NVIDIA GPU driver is not present.
* **Elementwise Kernels:** Optimized implementations for addition, subtraction, multiplication, and division.
* **Activation Functions:** Native implementations of ReLU, Sigmoid, GeLU, and Tanh.
* **Reduction Operations:** Global tensor sum and mean reductions.
* **Zero-Copy GPU Buffer Lifecycle:** Memory allocations managed through `forge_gpu_malloc`, `forge_gpu_free`, `forge_gpu_memcpy_to_device`, and `forge_gpu_memcpy_to_host`.

### Phase 2: Tensor Intermediate Representation (IR) & Tracing
* **SSA Graph Representation:** Pure functional Static Single Assignment graph structure consisting of `Graph`, `Node`, and `Value` classes. Values maintain unique SSA names (`%0`, `%1`), shapes, and constant data references.
* **OpKind Enumeration:** Strongly typed operator categories covering matrix multiplication, elementwise arithmetic, activations, and reductions.
* **Tracing Engine:** `forgecc.trace(fn, *example_inputs)` executes arbitrary Python callables with symbolic `TracerTensor` wrappers, intercepting operations to construct the SSA graph.
* **JIT Compiler:** `forgecc.compile(graph, target="gpu")` packages the SSA graph into an executable `CompiledGraph` callable.

### Phase 3: Compiler Optimization Passes
* **PassManager Architecture:** Pipeline coordinator running sequential graph transformation passes.
* **Constant Folding Pass (`ConstantFoldingPass`):** Evaluates subgraphs composed entirely of compile-time constants, replacing multi-node computations with pre-computed constant tensors.
* **Algebraic Simplification Pass (`AlgebraicSimplificationPass`):** Eliminates mathematical identities such as $x + 0 \to x$, $0 + x \to x$, $x \times 1 \to x$, and $1 \times x \to x$.
* **Common Subexpression Elimination Pass (`CommonSubexpressionEliminationPass`):** Identifies identical computations across the graph and merges redundant operations into a single shared value.
* **Dead Code Elimination Pass (`DeadCodeEliminationPass`):** Performs backward liveness analysis starting from registered graph outputs, pruning unused nodes and constants.

### Phase 4: CUDA CodeGen Backend & Zero-SDK Runtime
* **PTX Assembly Generation (`CUDACodeGen`):** Directly emits Parallel Thread Execution (PTX) assembly targeting NVIDIA virtual architectures (`.target sm_50`, `.address_size 64`).
* **Direct Driver JIT (`nvcuda.dll` / `libcuda.so`):** Dynamically loads the NVIDIA user-mode driver API at runtime via `LoadLibraryA` / `dlopen`. Resolves functions including `cuInit`, `cuDeviceGet`, `cuCtxCreate`, `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`, `cuMemAlloc`, and `cuMemcpyHtoD`.
* **Zero CUDA SDK Footprint:** Operates independently without requiring CUDA Toolkit installations or `nvcc` compilers.

### Phase 5: Advanced Operator Fusion
* **Epilogue & Residual Fusion (`matmul_add_relu_residual`):** Fuses matrix multiplication, vector bias addition, ReLU activation, and skip-connection residual addition into a single kernel:
  $$\text{Output} = \text{ReLU}(A \cdot B + \text{bias}) + \text{residual}$$
  Keeps intermediate accumulation values in hardware registers, eliminating global memory roundtrips.
* **Fused Elementwise Arithmetic (`add_relu`):** Combines vector addition and non-linear activation in a single pass.
* **Fusion Optimization Pass (`OperatorFusionPass`):** Automatically scans IR graph topologies to match combinable node patterns and replace them with fused execution nodes.

### Phase 6: GPU Memory Pool Optimization
* **Bucketed Memory Pool (`GPUMemoryPool`):** Implements a power-of-two memory bucket cache ($256, 512, 1024, \dots, 2^k$ bytes).
* **Allocation Overhead Elimination:** Intercepts deallocations to retain allocated GPU pointers in bucketed free lists. Subsequent allocations reuse cached pointers without calling `cuMemAlloc`.
* **Introspection API:** `forgecc.memory_stats()` reports active allocation counts and cached buffer metrics; `forgecc.memory_clear()` releases pooled VRAM.

### Phase 7: Hardware-Aware Auto-Tuning
* **Driver Device Querying:** Interrogates physical GPU properties dynamically via `cuDeviceGetAttribute` and `cuMemGetInfo` (Streaming Multiprocessor counts, max threads per block, warp size, shared memory capacity, and compute capability).
* **Elementwise Kernel Tuning (`auto_tune_elementwise`):** Computes optimal grid and thread block dimensions based on element count and GPU occupancy.
* **Matrix Multiplication Tuning (`auto_tune_matmul`):** Selects 2D thread block tiling parameters ($16 \times 16$ thread grids) and tile counts matched to dimension boundaries.

### Phase 8: ONNX Model Import Frontend
* **Zero-Dependency Protobuf Parser (`python/forgecc/onnx_frontend.py`):** Self-contained binary Protocol Buffer decoder implementing varint decoding, 64-bit/32-bit fixed fields, and length-delimited byte stream parsing.
* **ONNX Graph Translation:** Extracts `NodeProto`, `TensorProto` (weights/initializers), and `ValueInfoProto` (inputs/outputs) to construct a ForgeCC SSA IR graph.
* **Model Compilation:** `forgecc.from_onnx(model_path)` and `forgecc.compile_onnx(model_path)` convert standard ONNX models into compiled execution graphs.

### Phase 9: PyTorch & Framework Adapters
* **Symbolic Graph Translation (`torch_fx_to_forgecc_ir`):** Translates PyTorch `torch.fx.GraphModule` symbolic execution traces into ForgeCC SSA graphs.
* **Constant Broadcasting:** Automatically broadcasts scalar operands and parameters to matching tensor dimensions during IR generation.
* **Module Compilation Decorator (`@compile_torch_module`):** Compiles standard `torch.nn.Module` classes for execution on ForgeCC backends.
* **Bidirectional Tensor Bridge:** `torch_to_forgecc` and `forgecc_to_torch` allow seamless data exchange between PyTorch tensors and ForgeCC native tensors.

### Phase 10: Autograd Engine & Training Pipeline
* **Reverse-Mode Differentiation Tape (`backward_tape`):** Constructs dynamic topological execution order from output loss to leaf parameters.
* **Differentiable Operators:** Custom backward routines for `MatMulBackward`, `AddBackward`, `SubBackward`, `MulBackward`, `DivBackward`, `ReluBackward`, `SigmoidBackward`, and `TanhBackward`.
* **Loss Functions:** `forgecc.mse_loss(pred, target)` computes mean squared error loss with analytical gradient tracking.
* **Native Optimizers (`forgecc.optim`):**
  - **`SGD`:** Supports learning rate, momentum velocity accumulation, and L2 weight decay.
  - **`Adam`:** Supports first and second moment tracking ($m_t, v_t$), bias correction, and numerical stability epsilon.
  - **`AdamW`:** Decoupled weight decay regularization.
  - **In-Place Updates:** Updates tensor memory in-place via native C++ (`cpu_sgd_step`, `cpu_adam_step`) and GPU kernel routines.

---

## 5. Usage Guide & Code Examples

### 5.1 Basic Tensor Arithmetic & GPU Execution

```python
import forgecc

a = forgecc.tensor([[1.0, 2.0], [3.0, 4.0]], device=forgecc.Device.GPU)
b = forgecc.tensor([[5.0, 6.0], [7.0, 8.0]], device=forgecc.Device.GPU)

c = a @ b
d = (c + 1.0).relu()

print(d.numpy())
```

### 5.2 Tracing and Compiling Functions

```python
import forgecc

def forward_network(x, w1, b1):
    h = (x @ w1 + b1).relu()
    return h

x_sample = forgecc.tensor([[1.0, 2.0]])
w_sample = forgecc.tensor([[0.5, -0.2], [0.1, 0.8]])
b_sample = forgecc.tensor([0.1, 0.2])

graph = forgecc.trace(forward_network, x_sample, w_sample, b_sample)
optimized_graph = graph.optimize()
compiled_fn = forgecc.CompiledGraph(optimized_graph, target="gpu")

result = compiled_fn(x_sample, w_sample, b_sample)
print(result.numpy())
```

### 5.3 Importing and Compiling ONNX Models

```python
import forgecc

compiled_onnx = forgecc.compile_onnx("model.onnx", target="gpu", optimize=True)

input_tensor = forgecc.tensor([[1.0, 2.0, 3.0, 4.0]])
output = compiled_onnx(input_tensor)
print(output.numpy())
```

### 5.4 Compiling PyTorch Modules

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

torch_model = NeuralNet()
example_input = torch.randn(1, 4)

compiled_model = forgecc.compile_torch_module(torch_model, example_input, target="gpu")
output = compiled_model(example_input)
print(output)
```

### 5.5 End-to-End Training Loop with Autograd & Adam

```python
import forgecc
import numpy as np

x = forgecc.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], requires_grad=False)
y_true = forgecc.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=False)

w1 = forgecc.tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
w2 = forgecc.tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)

optimizer = forgecc.optim.Adam([w1, w2], lr=0.01)

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

## 6. Key Design Decisions & Architectural Innovations

| Feature | Traditional ML Frameworks | ForgeCC Compiler |
| :--- | :--- | :--- |
| **CUDA Dependency** | Requires CUDA Toolkit (3GB - 10GB) | Zero SDK requirement (dynamic `nvcuda.dll` driver loading) |
| **Compilation** | Heavyweight C++ compiler dependencies | Standalone PTX assembly generator emitting direct GPU bytecode |
| **ONNX Parsing** | Requires `protobuf` and `onnx` Python packages | Native zero-dependency binary Protocol Buffer wire-format parser |
| **Operator Fusion** | Separate kernel launches or runtime JIT overhead | Direct register-level multi-input epilogue & residual fusion |
| **GPU Memory** | Frequent driver allocation requests | Bucketed power-of-two VRAM memory pool with instant buffer reuse |
| **Framework Interop** | Siloed ecosystem | Native PyTorch FX adapter and standalone Python API |
