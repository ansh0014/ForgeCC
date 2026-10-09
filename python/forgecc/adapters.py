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

        elif node.op == "get_attr":
            sub_obj = gm
            for part in str(node.target).split("."):
                sub_obj = getattr(sub_obj, part)
            if isinstance(sub_obj, torch.Tensor):
                c_val = g.add_constant(sub_obj.detach().cpu().numpy())
                env[node.name] = c_val

        elif node.op == "call_module":
            submod = gm.get_submodule(str(node.target))
            if isinstance(submod, nn.Linear):
                inp_val = env[node.args[0].name]
                w_arr = submod.weight.detach().cpu().numpy().T
                w_val = g.add_constant(w_arr)
                out_shape = (inp_val.shape[0], w_arr.shape[1]) if len(inp_val.shape) == 2 else inp_val.shape
                mm_val = g.add_node(OpKind.MATMUL, [inp_val, w_val], out_shape)
                if submod.bias is not None:
                    b_arr = submod.bias.detach().cpu().numpy()
                    b_val = g.add_constant(b_arr)
                    add_val = g.add_node(OpKind.ADD, [mm_val, b_val], out_shape)
                    env[node.name] = add_val
                else:
                    env[node.name] = mm_val
            elif isinstance(submod, nn.ReLU):
                inp_val = env[node.args[0].name]
                env[node.name] = g.add_node(OpKind.RELU, [inp_val], inp_val.shape)
            elif isinstance(submod, nn.Sigmoid):
                inp_val = env[node.args[0].name]
                env[node.name] = g.add_node(OpKind.SIGMOID, [inp_val], inp_val.shape)
            elif isinstance(submod, nn.Tanh):
                inp_val = env[node.args[0].name]
                env[node.name] = g.add_node(OpKind.TANH, [inp_val], inp_val.shape)
            elif isinstance(submod, nn.GELU):
                inp_val = env[node.args[0].name]
                env[node.name] = g.add_node(OpKind.GELU, [inp_val], inp_val.shape)

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

            if "linear" in target_name.lower():
                if len(mapped_inputs) >= 2:
                    w_val = mapped_inputs[1]
                    if w_val.is_constant():
                        w_arr = w_val.const_data
                        if len(w_arr.shape) == 2 and len(mapped_inputs[0].shape) == 2 and mapped_inputs[0].shape[1] == w_arr.shape[1]:
                            w_val = g.add_constant(w_arr.T)
                            mapped_inputs[1] = w_val
                    s0 = mapped_inputs[0].shape
                    s1 = mapped_inputs[1].shape
                    if len(s0) == 2 and len(s1) == 2:
                        out_shape = (s0[0], s1[1])
                    mm_val = g.add_node(OpKind.MATMUL, [mapped_inputs[0], mapped_inputs[1]], out_shape)
                    if len(mapped_inputs) >= 3:
                        add_val = g.add_node(OpKind.ADD, [mm_val, mapped_inputs[2]], out_shape)
                        env[node.name] = add_val
                    else:
                        env[node.name] = mm_val
                    continue

            op_kind = OpKind.ADD
            if any(k in target_name.lower() for k in ["add", "operator.add"]):
                op_kind = OpKind.ADD
            elif any(k in target_name.lower() for k in ["sub", "operator.sub"]):
                op_kind = OpKind.SUB
            elif any(k in target_name.lower() for k in ["mul", "operator.mul"]):
                op_kind = OpKind.MUL
            elif any(k in target_name.lower() for k in ["div", "truediv"]):
                op_kind = OpKind.DIV
            elif any(k in target_name.lower() for k in ["mm", "matmul", "bmm", "gemm"]):
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
