# ForgeCC: High-Performance ML Compiler & Universal GPU Execution Engine

ForgeCC is a framework-independent machine learning compiler and high-throughput execution runtime designed to compile and execute computation graphs across multi-core CPUs and NVIDIA GPUs with zero external CUDA Toolkit dependencies.

---

## Key Highlights

- **Zero-SDK CUDA Acceleration:** Uses a direct dynamic loader for the NVIDIA display driver (`nvcuda.dll` on Windows, `libcuda.so` on Linux) with embedded and JIT-compiled PTX assembly. Runs on any system with NVIDIA drivers installed without requiring the CUDA Toolkit or `nvcc`.
- **Static Single Assignment (SSA) Tensor IR:** Represents computation graphs as pure functional SSA graphs supporting graph tracing, serialization, and hardware-targeted code generation.
- **Compiler Optimization Pipeline:** Modular `PassManager` executing Constant Folding, Algebraic Simplification ($x + 0 \to x$, $x \times 1 \to x$), Common Subexpression Elimination (CSE), and Dead Code Elimination (DCE).
- **Register-Level Operator Fusion:** Epilogue and residual fusion (`matmul_add_relu_residual` and `add_relu`) executing multi-operator sequences in hardware registers to eliminate intermediate global memory traffic.
- **Bucketed GPU Memory Pool:** Power-of-two memory bucket recycling (`GPUMemoryPool`) eliminating driver allocation latencies.
- **Hardware-Aware Auto-Tuning:** Dynamic driver introspection querying Streaming Multiprocessor counts, shared memory, and compute capability to tune thread block and grid launch dimensions.
- **Zero-Dependency ONNX Frontend:** Standalone binary Protocol Buffer decoder importing standard multi-layer ONNX models directly into ForgeCC IR.
- **PyTorch FX Integration:** Direct translation of `torch.fx.GraphModule` traces via `@compile_torch_module` and `forgecc_backend`.
- **Reverse-Mode Autograd & Optimizers:** Topological automatic differentiation tape with analytical gradients for tensor operations and MSE loss, alongside native in-place optimizers (`SGD`, `Adam`, `AdamW`).

---

## Architecture Overview

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
|  |   4. Operator & Residual Fusion Pass                                  |  |
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

## Verification & Test Results (14 / 14 Categories Passed)

ForgeCC is verified against a 14-category testing plan ([`docs/test_results.md`](docs/test_results.md) and [`validation_report.md`](validation_report.md)).

| Test ID | Category | Status | Backend | Result Details |
| :--- | :--- | :--- | :--- | :--- |
| **T01** | Tensor Arithmetic Correctness | **PASS** | CPU & GPU | 11 operators validated vs NumPy (`rtol=1e-5`, `atol=1e-6`). |
| **T02** | Shapes & Edge Cases | **PASS** | CPU & GPU | Non-square, prime, and non-divisible dimensions verified. |
| **T03** | SSA Intermediate Representation | **PASS** | Native IR | Acyclic dependency graph construction with unique SSA values. |
| **T04** | Compiler Optimization Passes | **PASS** | Native IR | Constant folding, algebraic simplification, CSE, and DCE verified. |
| **T05** | PTX Generation & GPU Execution | **PASS** | NVIDIA GPU | Virtual ISA PTX emitted targeting `sm_50`, executed via driver. |
| **T06** | Operator Fusion | **PASS** | GPU & CPU | Epilogue residual fusion $\text{ReLU}(A B + b) + R$ matched NumPy ($0.0$ diff). |
| **T07** | GPU Memory Pool | **PASS** | NVIDIA GPU | Bucketed power-of-two memory allocation recycling verified. |
| **T08** | Hardware-Aware Auto-Tuning | **PASS** | Driver API | Dynamic SM query and launch configuration selection verified. |
| **T09** | ONNX Frontend | **PASS** | Protobuf Engine | Zero-dependency binary Protobuf reader translated models to IR. |
| **T10** | PyTorch FX Adapter | **PASS** | PyTorch / FX | `torch.fx` graph capture & `@compile_torch_module` matched eager output. |
| **T11** | Automatic Differentiation | **PASS** | Autograd Tape | Reverse-mode analytical gradients verified against closed-form calculus. |
| **T12** | Optimizer Correctness | **PASS** | C++ / GPU | In-place parameter updates verified for SGD, Adam, and AdamW. |
| **T13** | Performance Benchmarking | **PASS** | GPU vs NumPy | Up to 4.01x speedup measured for matrix operations. |
| **T14** | Stress & Regression Testing | **PASS** | Native / Driver | 90+ repeated compilation, execution, and memory pool cycles clean. |

---

## Installation & Build

### Prerequisites

- **Operating System:** Windows 10/11 or Linux
- **C++ Compiler:** GCC / G++ (C++17 support) or MSVC
- **CMake:** 3.20+
- **Python:** 3.9+
- **NVIDIA GPU:** Any Kepler / Maxwell / Pascal / Volta / Turing / Ampere / Ada / Blackwell GPU with display drivers installed.

### Native Runtime Build

```cmd
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --target forgecc_rt_shared
```

### Python Package Installation

```cmd
py -m pip install -e ./python --no-build-isolation
```

---

## Usage Examples

### 1. Basic Tensor Arithmetic & GPU Execution

```python
import forgecc

a = forgecc.tensor([[1.0, 2.0], [3.0, 4.0]], device=forgecc.Device.GPU)
b = forgecc.tensor([[5.0, 6.0], [7.0, 8.0]], device=forgecc.Device.GPU)

c = a @ b
d = (c + 1.0).relu()

print(d.numpy())
```

### 2. Tracing and Compiling Python Functions

```python
import forgecc

def model_fn(x, w, b):
    return (x @ w + b).relu()

x = forgecc.tensor([[1.0, 2.0]])
w = forgecc.tensor([[0.5, -0.2], [0.1, 0.8]])
b = forgecc.tensor([0.1, 0.2])

graph = forgecc.trace(model_fn, x, w, b)
opt_graph = graph.optimize()
compiled_model = forgecc.CompiledGraph(opt_graph, target="gpu")

output = compiled_model(x, w, b)
print(output.numpy())
```

### 3. Compiling PyTorch Modules

```python
import torch
import torch.nn as nn
import forgecc

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

model = NeuralNet()
example_input = torch.randn(1, 4)

compiled_model = forgecc.compile_torch_module(model, example_input, target="gpu")
output = compiled_model(example_input)
print(output)
```

### 4. Neural Network Training Loop with Autograd & Adam

```python
import forgecc
import numpy as np

x = forgecc.tensor([[1.0, 2.0], [2.0, 3.0]], requires_grad=False)
y_true = forgecc.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=False)

w = forgecc.tensor(np.random.randn(2, 2).astype(np.float32), requires_grad=True)
optimizer = forgecc.optim.Adam([w], lr=0.01)

for epoch in range(20):
    optimizer.zero_grad()
    
    y_pred = (x @ w).relu()
    loss = forgecc.mse_loss(y_pred, y_true)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {float(loss.numpy().item()):.6f}")
```

---

## Documentation

- [`whole.md`](whole.md) — Comprehensive technical architecture, compiler pipeline, and subsystem guide.
- [`working.md`](working.md) — 10-Phase roadmap milestone tracking dashboard.
- [`validation_report.md`](validation_report.md) — Detailed execution log of the T01–T14 test suite.
- [`docs/test_results.md`](docs/test_results.md) — Official test verification data and benchmarks.

---

## License

ForgeCC is released under the MIT License.
