import numpy as np
from typing import List, Dict, Any, Callable, Optional, Union
from .core import Tensor, Device, is_gpu_available
from .ir import Graph, OpKind, Value, PassManager, CompiledGraph

try:
    import torch
    import torch.nn as nn
    import torch.fx as fx
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    fx = None
    _TORCH_AVAILABLE = False

def is_torch_available() -> bool:
    return _TORCH_AVAILABLE

def torch_to_forgecc(t: Any, device: Optional[int] = None) -> Tensor:
    if not _TORCH_AVAILABLE or not isinstance(t, torch.Tensor):
        if isinstance(t, Tensor):
            return t
        return Tensor(t)
    arr = t.detach().cpu().numpy()
    if device is None:
        device = Device.GPU if (t.is_cuda and is_gpu_available()) else Device.CPU
    return Tensor(arr, device=device)

def forgecc_to_torch(t: Tensor) -> Any:
    if not _TORCH_AVAILABLE:
        return t.numpy()
    arr = t.numpy()
    torch_t = torch.from_numpy(arr)
    if t.device == Device.GPU and torch.cuda.is_available():
        return torch_t.cuda()
    return torch_t

def torch_fx_to_forgecc_ir(gm: Any, example_inputs: List[Any]) -> Graph:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to convert torch.fx GraphModule to ForgeCC IR.")

    g = Graph(getattr(gm, "__name__", "torch_fx_graph"))
    env: Dict[str, Value] = {}

    inp_idx = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if inp_idx < len(example_inputs):
                ex = example_inputs[inp_idx]
                shape = tuple(ex.shape) if hasattr(ex, "shape") else (1,)
            else:
                shape = (1,)
            val = g.add_input(node.target, shape)
            env[node.name] = val
            inp_idx += 1

        elif node.op in ("call_function", "call_method"):
            target_name = str(node.target)
            args = node.args

            known_shape = None
            for arg in args:
                if isinstance(arg, fx.Node) and arg.name in env:
                    known_shape = env[arg.name].shape
                    break

            mapped_inputs: List[Value] = []
            for arg in args:
                if isinstance(arg, fx.Node) and arg.name in env:
                    mapped_inputs.append(env[arg.name])
                elif isinstance(arg, (int, float)):
                    if known_shape is not None:
                        c_arr = np.full(known_shape, float(arg), dtype=np.float32)
                    else:
                        c_arr = np.array([arg], dtype=np.float32)
                    c_val = g.add_constant(c_arr)
                    mapped_inputs.append(c_val)
                elif isinstance(arg, torch.Tensor):
                    c_arr = arg.detach().cpu().numpy()
                    if known_shape is not None and c_arr.size == 1 and np.prod(known_shape) > 1:
                        c_arr = np.full(known_shape, float(c_arr.item()), dtype=np.float32)
                    c_val = g.add_constant(c_arr)
                    mapped_inputs.append(c_val)

            out_shape = (1,)
            if mapped_inputs:
                out_shape = mapped_inputs[0].shape

            op_kind = OpKind.ADD
            if any(k in target_name.lower() for k in ["add", "operator.add"]):
                op_kind = OpKind.ADD
            elif any(k in target_name.lower() for k in ["sub", "operator.sub"]):
                op_kind = OpKind.SUB
            elif any(k in target_name.lower() for k in ["mul", "operator.mul"]):
                op_kind = OpKind.MUL
            elif any(k in target_name.lower() for k in ["div", "truediv"]):
                op_kind = OpKind.DIV
            elif any(k in target_name.lower() for k in ["mm", "matmul", "linear", "bmm", "gemm"]):
                op_kind = OpKind.MATMUL
                if len(mapped_inputs) >= 2:
                    s0 = mapped_inputs[0].shape
                    s1 = mapped_inputs[1].shape
                    if len(s0) == 2 and len(s1) == 2:
                        out_shape = (s0[0], s1[1])
            elif "relu" in target_name.lower():
                op_kind = OpKind.RELU
            elif "sigmoid" in target_name.lower():
                op_kind = OpKind.SIGMOID
            elif "gelu" in target_name.lower():
                op_kind = OpKind.GELU
            elif "tanh" in target_name.lower():
                op_kind = OpKind.TANH
            elif "sum" in target_name.lower():
                op_kind = OpKind.SUM
                out_shape = (1,)
            elif "mean" in target_name.lower():
                op_kind = OpKind.MEAN
                out_shape = (1,)

            out_val = g.add_node(op_kind, mapped_inputs, out_shape)
            env[node.name] = out_val

        elif node.op == "output":
            outputs = node.args[0]
            if isinstance(outputs, (list, tuple)):
                for o in outputs:
                    if isinstance(o, fx.Node) and o.name in env:
                        g.register_output(env[o.name])
            elif isinstance(outputs, fx.Node) and outputs.name in env:
                g.register_output(env[outputs.name])

    return g

def forgecc_backend(model: Any, example_inputs: List[Any]) -> Callable:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to use forgecc_backend with torch.compile.")

    if isinstance(model, fx.GraphModule):
        ir_graph = torch_fx_to_forgecc_ir(model, example_inputs)
        opt_graph = ir_graph.optimize()
        compiled = CompiledGraph(opt_graph, target="gpu" if is_gpu_available() else "cpu")

        def compiled_forward(*args):
            f_args = [torch_to_forgecc(a) for a in args]
            res = compiled(*f_args)
            if isinstance(res, (tuple, list)):
                return tuple(forgecc_to_torch(r) for r in res)
            return forgecc_to_torch(res)

        return compiled_forward
    else:
        return model

def compile_torch_module(module: Any, *example_inputs: Any, target: str = "gpu", optimize: bool = True) -> Callable:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to compile a torch.nn.Module.")

    if isinstance(module, nn.Module):
        traced_gm = fx.symbolic_trace(module)
        ir_graph = torch_fx_to_forgecc_ir(traced_gm, list(example_inputs))
        if optimize:
            ir_graph = ir_graph.optimize()
        compiled = CompiledGraph(ir_graph, target=target)

        def callable_module(*args):
            f_args = [torch_to_forgecc(a) for a in args]
            res = compiled(*f_args)
            if isinstance(res, (tuple, list)):
                return tuple(forgecc_to_torch(r) for r in res)
            return forgecc_to_torch(res)

        return callable_module
    else:
        raise TypeError("Object passed to compile_torch_module must be an instance of torch.nn.Module.")
