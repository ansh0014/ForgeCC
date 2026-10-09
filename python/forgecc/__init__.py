from .core import (
    Device,
    is_gpu_available,
    sync,
    Tensor,
    tensor,
    matmul,
    add,
    sub,
    mul,
    div,
    relu,
    sigmoid,
    gelu,
    tanh,
    sum_tensor as sum,
    mean_tensor as mean,
    matmul_relu,
    matmul_add,
    matmul_add_relu,
    matmul_add_relu_residual,
    add_relu,
    memory_stats,
    memory_clear,
    get_device_properties,
    auto_tune_elementwise,
    auto_tune_matmul
)

from .ir import (
    OpKind,
    Value,
    Node,
    Graph,
    Pass,
    PassManager,
    ConstantFoldingPass,
    AlgebraicSimplificationPass,
    CommonSubexpressionEliminationPass,
    OperatorFusionPass,
    DeadCodeEliminationPass,
    CUDACodeGen,
    trace,
    compile,
    CompiledGraph
)

from .onnx_frontend import (
    from_onnx,
    compile_onnx
)

from .adapters import (
    is_torch_available,
    torch_to_forgecc,
    forgecc_to_torch,
    torch_fx_to_forgecc_ir,
    forgecc_backend,
    compile_torch_module
)

from .autograd import (
    mse_loss,
    backward_tape
)

from .optim import (
    Optimizer,
    SGD,
    Adam,
    AdamW
)

from . import optim
from . import adapters
from . import autograd

__version__ = "0.2.5"
__all__ = [
    "Device",
    "is_gpu_available",
    "sync",
    "Tensor",
    "tensor",
    "matmul",
    "add",
    "sub",
    "mul",
    "div",
    "relu",
    "sigmoid",
    "gelu",
    "tanh",
    "sum",
    "mean",
    "matmul_relu",
    "matmul_add",
    "matmul_add_relu",
    "matmul_add_relu_residual",
    "add_relu",
    "memory_stats",
    "memory_clear",
    "get_device_properties",
    "auto_tune_elementwise",
    "auto_tune_matmul",
    "from_onnx",
    "compile_onnx",
    "OpKind",
    "Value",
    "Node",
    "Graph",
    "Pass",
    "PassManager",
    "ConstantFoldingPass",
    "AlgebraicSimplificationPass",
    "CommonSubexpressionEliminationPass",
    "OperatorFusionPass",
    "DeadCodeEliminationPass",
    "CUDACodeGen",
    "trace",
    "compile",
    "CompiledGraph",
    "is_torch_available",
    "torch_to_forgecc",
    "forgecc_to_torch",
    "torch_fx_to_forgecc_ir",
    "forgecc_backend",
    "compile_torch_module",
    "mse_loss",
    "backward_tape",
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "optim",
    "adapters",
    "autograd"
]
