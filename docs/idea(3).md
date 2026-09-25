# ForgeCC — Project Idea

## 1. Project Overview

**ForgeCC** is a custom C++ compiler built using the LLVM compiler infrastructure.

The goal is to design a compiler that understands ordinary C++ programs while adding **GPU-aware compilation capabilities** for computationally intensive workloads, especially machine-learning and tensor-based computations.

ForgeCC is not intended to replace LLVM. LLVM is the core compiler infrastructure, while ForgeCC provides its own frontend extensions, analysis, intermediate representations, optimization passes, and GPU-oriented compilation logic.

---

## 2. Problem Statement

Modern C++ compilers are excellent at generating CPU machine code, but ML workloads contain computation patterns such as:

- Matrix multiplication
- Tensor operations
- Convolution
- Vector operations
- Reduction operations

These operations can benefit significantly from GPU execution.

A conventional compiler does not necessarily understand that a particular section of a C++ program represents an ML workload or that it would benefit from GPU execution.

ForgeCC aims to introduce compiler-level awareness of these workloads.

---

## 3. Vision

The long-term vision is:

> **Build a compiler that can analyze a C++ program, recognize GPU-suitable computational workloads, optimize them, and generate code capable of executing those workloads on the available hardware.**

The compiler should make decisions based on program characteristics rather than simply sending everything to the GPU.

---

## 4. Core Goals

### Compiler

- Build a real compiler pipeline in C++.
- Use LLVM as the primary compiler infrastructure.
- Understand lexical, syntactic, and semantic compilation stages.
- Generate LLVM IR.
- Use LLVM optimization and code-generation infrastructure.

### GPU Awareness

- Detect computationally intensive operations.
- Identify GPU-suitable workloads.
- Support explicit and eventually automatic GPU offloading.
- Generate GPU-oriented code through a dedicated backend.

### ML Awareness

The compiler should eventually understand concepts such as:

- Tensors
- Tensor shapes
- Matrix multiplication
- Convolution
- Element-wise operations
- Reductions
- Neural-network computation graphs

### Optimization

Potential compiler optimizations include:

- Constant folding
- Dead code elimination
- Operator fusion
- Loop optimization
- Vectorization
- Tensor-layout optimization
- Memory planning
- Device placement

---

## 5. What ForgeCC Is NOT

ForgeCC is **not**:

- A machine-learning framework.
- A replacement for PyTorch.
- A replacement for LLVM.
- A GPU driver.
- A CUDA replacement in the first version.
- A general-purpose compiler that attempts to implement every architecture from scratch.

LLVM will provide the lower-level compiler infrastructure.

---

## 6. Basic Compilation Concept

A simplified ForgeCC pipeline is:

```text
C++ Source
    |
    v
Frontend
    |
    v
AST
    |
    v
Semantic Analysis
    |
    v
ForgeCC Analysis / IR
    |
    +----------------------+
    |                      |
    v                      v
Normal C++             ML / Tensor
Workload               Workload
    |                      |
    |                      v
    |                 GPU Analysis
    |                      |
    +----------+-----------+
               |
               v
        Optimization
               |
               v
           LLVM IR
               |
        +------+------+
        |             |
        v             v
       CPU           GPU
     Backend       Backend
        |             |
        v             v
      CPU          GPU Code
     Machine       / Kernel
       Code
```

---

## 7. Example

A future ForgeCC program could contain a computation such as:

```cpp
Tensor<float> A;
Tensor<float> B;

auto C = matmul(A, B);
auto D = relu(C);
```

ForgeCC should eventually be able to recognize:

```text
MatMul
   |
   v
ReLU
```

and determine that this computation is suitable for GPU execution.

It could then perform optimizations such as:

```text
MatMul + ReLU
       |
       v
Fused GPU operation
```

The exact GPU code-generation mechanism will be developed incrementally.

---

## 8. CPU vs GPU Decision

ForgeCC should eventually have a device-selection mechanism.

```text
                Operation
                    |
                    v
               Cost Model
              /                       /                      CPU              GPU
           |                |
           v                v
       CPU Code         GPU Code
```

The decision can consider:

- Number of operations
- Tensor size
- Memory requirements
- CPU/GPU transfer cost
- GPU kernel-launch overhead
- Available GPU resources
- Data type
- Operation type

The objective is not:

> "Always use the GPU."

The objective is:

> **Choose an execution strategy based on the characteristics of the workload.**

---

## 9. Why LLVM?

LLVM provides mature infrastructure for:

- Intermediate representation
- Optimization passes
- Static analysis
- Target-independent transformations
- Instruction selection
- Register allocation
- Machine-code generation
- Debug information
- Multiple CPU targets

ForgeCC can therefore concentrate on the parts that make it unique:

```text
ML/Tensor Analysis
        +
GPU-Aware Optimization
        +
Device Placement
        +
GPU Code Generation
```

---

## 10. Initial Scope

The first version will intentionally be small.

### V1

Support:

- Basic C++ compilation
- LLVM IR generation
- Tensor abstraction
- Matrix multiplication
- Basic element-wise operations
- GPU workload identification
- One GPU target
- Basic GPU execution

### Later Versions

Potential additions:

- Operator fusion
- Automatic device placement
- Memory planning
- Tensor layout optimization
- Mixed precision
- Quantization
- Neural-network graph optimization
- Multiple GPU vendors
- Better cost models

---

## 11. Development Philosophy

ForgeCC will be developed incrementally.

```text
Compiler Fundamentals
        |
        v
LLVM Integration
        |
        v
Custom IR
        |
        v
Tensor Representation
        |
        v
ML Analysis
        |
        v
Optimization
        |
        v
GPU Backend
        |
        v
Runtime
```

Each stage should produce a working artifact before moving to the next stage.

---

## 12. Final Goal

The final system should demonstrate that a custom compiler can combine:

```text
C++
+
LLVM
+
Compiler Optimization
+
Tensor/ML Awareness
+
GPU Code Generation
+
Hardware-Aware Execution
```

to produce optimized programs capable of using both CPU and GPU resources.
