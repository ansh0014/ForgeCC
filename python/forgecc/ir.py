from enum import Enum
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from .core import (
    Device,
    Tensor,
    tensor,
    is_gpu_available,
    matmul,
    add,
    sub,
    mul,
    div,
    relu,
    sigmoid,
    gelu,
    tanh,
    sum_tensor,
    mean_tensor,
    matmul_relu,
    matmul_add,
    matmul_add_relu
)

class OpKind(Enum):
    INPUT = "input"
    CONSTANT = "constant"
    MATMUL = "matmul"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    RELU = "relu"
    SIGMOID = "sigmoid"
    GELU = "gelu"
    TANH = "tanh"
    SUM = "sum"
    MEAN = "mean"
    FUSED_MATMUL_RELU = "fused_matmul_relu"
    FUSED_MATMUL_ADD = "fused_matmul_add"
    FUSED_MATMUL_ADD_RELU = "fused_matmul_add_relu"

class Value:
    def __init__(self, name: str, shape: tuple, dtype: str = "float32", device: str = "cpu"):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.producer = None
        self.consumers: List['Node'] = []

    def __repr__(self):
        return f"{self.name}: Tensor{list(self.shape)}, {self.dtype}"

class Node:
    def __init__(self, op_kind: OpKind, inputs: List[Value], outputs: List[Value], name: str = ""):
        self.op_kind = op_kind
        self.inputs = inputs
        self.outputs = outputs
        self.name = name if name else op_kind.value

        for inp in inputs:
            inp.consumers.append(self)
        for out in outputs:
            out.producer = self

    def __repr__(self):
        outs_str = ", ".join(v.name for v in self.outputs)
        inps_str = ", ".join(v.name for v in self.inputs)
        return f"{outs_str} = {self.op_kind.value}({inps_str})"

class Graph:
    def __init__(self, name: str = "forgecc_graph"):
        self.name = name
        self.inputs: List[Value] = []
        self.outputs: List[Value] = []
        self.nodes: List[Node] = []
        self._val_counter = 0

    def new_value(self, shape: tuple, dtype: str = "float32", prefix: str = "%") -> Value:
        v = Value(f"{prefix}{self._val_counter}", shape, dtype)
        self._val_counter += 1
        return v

    def add_input(self, name: str, shape: tuple, dtype: str = "float32") -> Value:
        v = Value(name, shape, dtype)
        self.inputs.append(v)
        return v

    def add_node(self, op_kind: OpKind, inputs: List[Value], out_shape: tuple) -> Value:
        out_v = self.new_value(out_shape)
        node = Node(op_kind, inputs, [out_v])
        self.nodes.append(node)
        return out_v

    def register_output(self, v: Value):
        self.outputs.append(v)

    def dump(self) -> str:
        lines = []
        inps = ", ".join(f"{v.name}: Tensor{list(v.shape)}, {v.dtype}" for v in self.inputs)
        lines.append(f"graph {self.name}({inps}) {{")
        for node in self.nodes:
            lines.append(f"  {node}")
        outs = ", ".join(v.name for v in self.outputs)
        lines.append(f"  return {outs}")
        lines.append("}")
        return "\n".join(lines)

    def optimize(self) -> 'Graph':
        opt_graph = Graph(f"{self.name}_optimized")
        val_map: Dict[str, Value] = {}

        for inp in self.inputs:
            v_new = opt_graph.add_input(inp.name, inp.shape, inp.dtype)
            val_map[inp.name] = v_new

        i = 0
        while i < len(self.nodes):
            node = self.nodes[i]

            if node.op_kind == OpKind.MATMUL and (i + 2 < len(self.nodes)):
                next_node = self.nodes[i + 1]
                next_next = self.nodes[i + 2]

                if next_node.op_kind == OpKind.ADD and next_node.inputs[0] == node.outputs[0]:
                    if next_next.op_kind == OpKind.RELU and next_next.inputs[0] == next_node.outputs[0]:
                        in0 = val_map[node.inputs[0].name]
                        in1 = val_map[node.inputs[1].name]
                        in2 = val_map[next_node.inputs[1].name]
                        out_v = opt_graph.add_node(OpKind.FUSED_MATMUL_ADD_RELU, [in0, in1, in2], next_next.outputs[0].shape)
                        val_map[next_next.outputs[0].name] = out_v
                        i += 3
                        continue

            if node.op_kind == OpKind.MATMUL and (i + 1 < len(self.nodes)):
                next_node = self.nodes[i + 1]
                if next_node.op_kind == OpKind.RELU and next_node.inputs[0] == node.outputs[0]:
                    in0 = val_map[node.inputs[0].name]
                    in1 = val_map[node.inputs[1].name]
                    out_v = opt_graph.add_node(OpKind.FUSED_MATMUL_RELU, [in0, in1], next_node.outputs[0].shape)
                    val_map[next_node.outputs[0].name] = out_v
                    i += 2
                    continue

                if next_node.op_kind == OpKind.ADD and next_node.inputs[0] == node.outputs[0]:
                    in0 = val_map[node.inputs[0].name]
                    in1 = val_map[node.inputs[1].name]
                    in2 = val_map[next_node.inputs[1].name]
                    out_v = opt_graph.add_node(OpKind.FUSED_MATMUL_ADD, [in0, in1, in2], next_node.outputs[0].shape)
                    val_map[next_node.outputs[0].name] = out_v
                    i += 2
                    continue

            mapped_inputs = [val_map[inp.name] for inp in node.inputs]
            out_v = opt_graph.add_node(node.op_kind, mapped_inputs, node.outputs[0].shape)
            val_map[node.outputs[0].name] = out_v
            i += 1

        for out in self.outputs:
            opt_graph.register_output(val_map[out.name])

        return opt_graph

class TracerTensor:
    def __init__(self, value: Value, graph: Graph):
        self.value = value
        self.graph = graph
        self.shape = value.shape

    def __matmul__(self, other):
        other_val = other.value if isinstance(other, TracerTensor) else self.graph.new_value(other.shape)
        out_shape = (self.shape[0], other.shape[1])
        out_val = self.graph.add_node(OpKind.MATMUL, [self.value, other_val], out_shape)
        return TracerTensor(out_val, self.graph)

    def __add__(self, other):
        other_val = other.value if isinstance(other, TracerTensor) else self.graph.new_value(other.shape)
        out_val = self.graph.add_node(OpKind.ADD, [self.value, other_val], self.shape)
        return TracerTensor(out_val, self.graph)

    def __sub__(self, other):
        other_val = other.value if isinstance(other, TracerTensor) else self.graph.new_value(other.shape)
        out_val = self.graph.add_node(OpKind.SUB, [self.value, other_val], self.shape)
        return TracerTensor(out_val, self.graph)

    def __mul__(self, other):
        other_val = other.value if isinstance(other, TracerTensor) else self.graph.new_value(other.shape)
        out_val = self.graph.add_node(OpKind.MUL, [self.value, other_val], self.shape)
        return TracerTensor(out_val, self.graph)

    def relu(self):
        out_val = self.graph.add_node(OpKind.RELU, [self.value], self.shape)
        return TracerTensor(out_val, self.graph)

    def sigmoid(self):
        out_val = self.graph.add_node(OpKind.SIGMOID, [self.value], self.shape)
        return TracerTensor(out_val, self.graph)

    def gelu(self):
        out_val = self.graph.add_node(OpKind.GELU, [self.value], self.shape)
        return TracerTensor(out_val, self.graph)

    def tanh(self):
        out_val = self.graph.add_node(OpKind.TANH, [self.value], self.shape)
        return TracerTensor(out_val, self.graph)

    def sum(self):
        out_val = self.graph.add_node(OpKind.SUM, [self.value], (1,))
        return TracerTensor(out_val, self.graph)

    def mean(self):
        out_val = self.graph.add_node(OpKind.MEAN, [self.value], (1,))
        return TracerTensor(out_val, self.graph)

def trace(fn: Callable, *example_inputs) -> Graph:
    g = Graph(getattr(fn, "__name__", "model_graph"))
    tracer_inputs = []
    for i, inp in enumerate(example_inputs):
        shape = inp.shape if hasattr(inp, "shape") else (1,)
        v = g.add_input(f"%arg{i}", shape)
        tracer_inputs.append(TracerTensor(v, g))

    res = fn(*tracer_inputs)
    if isinstance(res, TracerTensor):
        g.register_output(res.value)
    elif isinstance(res, (tuple, list)):
        for r in res:
            if isinstance(r, TracerTensor):
                g.register_output(r.value)
    return g

class CompiledGraph:
    def __init__(self, graph: Graph, target: str = "gpu"):
        self.graph = graph
        self.target = target
        self.device = Device.GPU if target.lower() == "gpu" and is_gpu_available() else Device.CPU

    def __call__(self, *args):
        env: Dict[str, Tensor] = {}
        for param, arg in zip(self.graph.inputs, args):
            if isinstance(arg, Tensor):
                env[param.name] = arg.to(self.device)
            else:
                env[param.name] = Tensor(arg, device=self.device)

        for node in self.graph.nodes:
            inps = [env[v.name] for v in node.inputs]
            if node.op_kind == OpKind.MATMUL:
                out = matmul(inps[0], inps[1])
            elif node.op_kind == OpKind.ADD:
                out = add(inps[0], inps[1])
            elif node.op_kind == OpKind.SUB:
                out = sub(inps[0], inps[1])
            elif node.op_kind == OpKind.MUL:
                out = mul(inps[0], inps[1])
            elif node.op_kind == OpKind.RELU:
                out = relu(inps[0])
            elif node.op_kind == OpKind.SIGMOID:
                out = sigmoid(inps[0])
            elif node.op_kind == OpKind.GELU:
                out = gelu(inps[0])
            elif node.op_kind == OpKind.TANH:
                out = tanh(inps[0])
            elif node.op_kind == OpKind.SUM:
                out = sum_tensor(inps[0])
            elif node.op_kind == OpKind.MEAN:
                out = mean_tensor(inps[0])
            elif node.op_kind == OpKind.FUSED_MATMUL_RELU:
                out = matmul_relu(inps[0], inps[1])
            elif node.op_kind == OpKind.FUSED_MATMUL_ADD:
                out = matmul_add(inps[0], inps[1], inps[2])
            elif node.op_kind == OpKind.FUSED_MATMUL_ADD_RELU:
                out = matmul_add_relu(inps[0], inps[1], inps[2])
            else:
                raise NotImplementedError(f"Unsupported OpKind: {node.op_kind}")

            env[node.outputs[0].name] = out

        results = [env[v.name] for v in self.graph.outputs]
        return results[0] if len(results) == 1 else tuple(results)

def compile(fn_or_graph, target: str = "gpu", optimize: bool = True) -> CompiledGraph:
    if isinstance(fn_or_graph, Graph):
        g = fn_or_graph
    else:
        raise ValueError("Pass traced Graph or use forgecc.compile with trace.")
    if optimize:
        g = g.optimize()
    return CompiledGraph(g, target=target)
