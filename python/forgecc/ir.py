from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
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
    matmul_add_relu,
    matmul_add_relu_residual,
    add_relu
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
    FUSED_ADD_RELU = "fused_add_relu"
    FUSED_MATMUL_RELU = "fused_matmul_relu"
    FUSED_MATMUL_ADD = "fused_matmul_add"
    FUSED_MATMUL_ADD_RELU = "fused_matmul_add_relu"
    FUSED_MATMUL_ADD_RELU_RESIDUAL = "fused_matmul_add_relu_residual"

class Value:
    def __init__(self, name: str, shape: tuple, dtype: str = "float32", const_data: Optional[np.ndarray] = None):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.const_data = const_data
        self.producer: Optional['Node'] = None
        self.consumers: List['Node'] = []

    def is_constant(self) -> bool:
        return self.const_data is not None

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
        self.constants: List[Value] = []
        self._val_counter = 0

    def new_value(self, shape: tuple, dtype: str = "float32", prefix: str = "%", const_data: Optional[np.ndarray] = None) -> Value:
        v = Value(f"{prefix}{self._val_counter}", shape, dtype, const_data)
        self._val_counter += 1
        if const_data is not None:
            self.constants.append(v)
        return v

    def add_input(self, name: str, shape: tuple, dtype: str = "float32") -> Value:
        v = Value(name, shape, dtype)
        self.inputs.append(v)
        return v

    def add_constant(self, data: np.ndarray, prefix: str = "%c") -> Value:
        arr = np.ascontiguousarray(data, dtype=np.float32)
        v = self.new_value(arr.shape, dtype="float32", prefix=prefix, const_data=arr)
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
        pm = PassManager()
        pm.add_pass(ConstantFoldingPass())
        pm.add_pass(AlgebraicSimplificationPass())
        pm.add_pass(CommonSubexpressionEliminationPass())
        pm.add_pass(OperatorFusionPass())
        pm.add_pass(DeadCodeEliminationPass())
        return pm.run(self)

class Pass:
    def run(self, graph: Graph) -> Graph:
        raise NotImplementedError

class PassManager:
    def __init__(self):
        self.passes: List[Pass] = []

    def add_pass(self, p: Pass):
        self.passes.append(p)

    def run(self, graph: Graph) -> Graph:
        current = graph
        for p in self.passes:
            current = p.run(current)
        return current

class ConstantFoldingPass(Pass):
    def run(self, graph: Graph) -> Graph:
        opt = Graph(graph.name)
        val_map: Dict[str, Value] = {}

        for inp in graph.inputs:
            val_map[inp.name] = opt.add_input(inp.name, inp.shape, inp.dtype)
        for c in graph.constants:
            val_map[c.name] = opt.add_constant(c.const_data, prefix=c.name.rstrip("0123456789"))

        for node in graph.nodes:
            all_const = all(val_map[inp.name].is_constant() for inp in node.inputs)
            if all_const and len(node.inputs) > 0:
                const_inputs = [val_map[inp.name].const_data for inp in node.inputs]
                if node.op_kind == OpKind.MATMUL:
                    res = np.matmul(const_inputs[0], const_inputs[1])
                elif node.op_kind == OpKind.ADD:
                    res = const_inputs[0] + const_inputs[1]
                elif node.op_kind == OpKind.SUB:
                    res = const_inputs[0] - const_inputs[1]
                elif node.op_kind == OpKind.MUL:
                    res = const_inputs[0] * const_inputs[1]
                elif node.op_kind == OpKind.RELU:
                    res = np.maximum(0, const_inputs[0])
                elif node.op_kind == OpKind.SIGMOID:
                    res = 1.0 / (1.0 + np.exp(-const_inputs[0]))
                else:
                    res = None

                if res is not None:
                    c_val = opt.add_constant(res)
                    val_map[node.outputs[0].name] = c_val
                    continue

            mapped_inps = [val_map[inp.name] for inp in node.inputs]
            out_v = opt.add_node(node.op_kind, mapped_inps, node.outputs[0].shape)
            val_map[node.outputs[0].name] = out_v

        for out in graph.outputs:
            opt.register_output(val_map[out.name])

        return opt

class AlgebraicSimplificationPass(Pass):
    def run(self, graph: Graph) -> Graph:
        opt = Graph(graph.name)
        val_map: Dict[str, Value] = {}

        for inp in graph.inputs:
            val_map[inp.name] = opt.add_input(inp.name, inp.shape, inp.dtype)
        for c in graph.constants:
            val_map[c.name] = opt.add_constant(c.const_data, prefix=c.name.rstrip("0123456789"))

        for node in graph.nodes:
            mapped_inps = [val_map[inp.name] for inp in node.inputs]
            if node.op_kind == OpKind.ADD and len(mapped_inps) == 2:
                if mapped_inps[1].is_constant() and np.all(mapped_inps[1].const_data == 0):
                    val_map[node.outputs[0].name] = mapped_inps[0]
                    continue
                if mapped_inps[0].is_constant() and np.all(mapped_inps[0].const_data == 0):
                    val_map[node.outputs[0].name] = mapped_inps[1]
                    continue

            if node.op_kind == OpKind.MUL and len(mapped_inps) == 2:
                if mapped_inps[1].is_constant() and np.all(mapped_inps[1].const_data == 1):
                    val_map[node.outputs[0].name] = mapped_inps[0]
                    continue
                if mapped_inps[0].is_constant() and np.all(mapped_inps[0].const_data == 1):
                    val_map[node.outputs[0].name] = mapped_inps[1]
                    continue

            out_v = opt.add_node(node.op_kind, mapped_inps, node.outputs[0].shape)
            val_map[node.outputs[0].name] = out_v

        for out in graph.outputs:
            opt.register_output(val_map[out.name])

        return opt

class CommonSubexpressionEliminationPass(Pass):
    def run(self, graph: Graph) -> Graph:
        opt = Graph(graph.name)
        val_map: Dict[str, Value] = {}
        expr_cache: Dict[Tuple[OpKind, Tuple[str, ...]], Value] = {}

        for inp in graph.inputs:
            val_map[inp.name] = opt.add_input(inp.name, inp.shape, inp.dtype)
        for c in graph.constants:
            val_map[c.name] = opt.add_constant(c.const_data, prefix=c.name.rstrip("0123456789"))

        for node in graph.nodes:
            mapped_inps = [val_map[inp.name] for inp in node.inputs]
            key = (node.op_kind, tuple(v.name for v in mapped_inps))

            if key in expr_cache:
                val_map[node.outputs[0].name] = expr_cache[key]
            else:
                out_v = opt.add_node(node.op_kind, mapped_inps, node.outputs[0].shape)
                expr_cache[key] = out_v
                val_map[node.outputs[0].name] = out_v

        for out in graph.outputs:
            opt.register_output(val_map[out.name])

        return opt

class OperatorFusionPass(Pass):
    def run(self, graph: Graph) -> Graph:
        opt = Graph(f"{graph.name}_fused")
        val_map: Dict[str, Value] = {}

        for inp in graph.inputs:
            val_map[inp.name] = opt.add_input(inp.name, inp.shape, inp.dtype)
        for c in graph.constants:
            val_map[c.name] = opt.add_constant(c.const_data, prefix=c.name.rstrip("0123456789"))

        i = 0
        while i < len(graph.nodes):
            node = graph.nodes[i]

            if node.op_kind == OpKind.MATMUL and (i + 3 < len(graph.nodes)):
                n1 = graph.nodes[i + 1]
                n2 = graph.nodes[i + 2]
                n3 = graph.nodes[i + 3]
                if n1.op_kind == OpKind.ADD and n1.inputs[0] == node.outputs[0]:
                    if n2.op_kind == OpKind.RELU and n2.inputs[0] == n1.outputs[0]:
                        if n3.op_kind == OpKind.ADD and n3.inputs[0] == n2.outputs[0]:
                            in_a = val_map[node.inputs[0].name]
                            in_b = val_map[node.inputs[1].name]
                            in_bias = val_map[n1.inputs[1].name]
                            in_res = val_map[n3.inputs[1].name]
                            out_v = opt.add_node(OpKind.FUSED_MATMUL_ADD_RELU_RESIDUAL, [in_a, in_b, in_bias, in_res], n3.outputs[0].shape)
                            val_map[n3.outputs[0].name] = out_v
                            i += 4
                            continue

            if node.op_kind == OpKind.MATMUL and (i + 2 < len(graph.nodes)):
                next_node = graph.nodes[i + 1]
                next_next = graph.nodes[i + 2]

                if next_node.op_kind == OpKind.ADD and next_node.inputs[0] == node.outputs[0]:
                    if next_next.op_kind == OpKind.RELU and next_next.inputs[0] == next_node.outputs[0]:
                        in0 = val_map[node.inputs[0].name]
                        in1 = val_map[node.inputs[1].name]
                        in2 = val_map[next_node.inputs[1].name]
                        out_v = opt.add_node(OpKind.FUSED_MATMUL_ADD_RELU, [in0, in1, in2], next_next.outputs[0].shape)
                        val_map[next_next.outputs[0].name] = out_v
                        i += 3
                        continue

            if node.op_kind == OpKind.MATMUL and (i + 1 < len(graph.nodes)):
                next_node = graph.nodes[i + 1]
                if next_node.op_kind == OpKind.RELU and next_node.inputs[0] == node.outputs[0]:
                    in0 = val_map[node.inputs[0].name]
                    in1 = val_map[node.inputs[1].name]
                    out_v = opt.add_node(OpKind.FUSED_MATMUL_RELU, [in0, in1], next_node.outputs[0].shape)
                    val_map[next_node.outputs[0].name] = out_v
                    i += 2
                    continue

                if next_node.op_kind == OpKind.ADD and next_node.inputs[0] == node.outputs[0]:
                    in0 = val_map[node.inputs[0].name]
                    in1 = val_map[node.inputs[1].name]
                    in2 = val_map[next_node.inputs[1].name]
                    out_v = opt.add_node(OpKind.FUSED_MATMUL_ADD, [in0, in1, in2], next_node.outputs[0].shape)
                    val_map[next_node.outputs[0].name] = out_v
                    i += 2
                    continue

            if node.op_kind == OpKind.ADD and (i + 1 < len(graph.nodes)):
                next_node = graph.nodes[i + 1]
                if next_node.op_kind == OpKind.RELU and next_node.inputs[0] == node.outputs[0]:
                    in0 = val_map[node.inputs[0].name]
                    in1 = val_map[node.inputs[1].name]
                    out_v = opt.add_node(OpKind.FUSED_ADD_RELU, [in0, in1], next_node.outputs[0].shape)
                    val_map[next_node.outputs[0].name] = out_v
                    i += 2
                    continue

            mapped_inputs = [val_map[inp.name] for inp in node.inputs]
            out_v = opt.add_node(node.op_kind, mapped_inputs, node.outputs[0].shape)
            val_map[node.outputs[0].name] = out_v
            i += 1

        for out in graph.outputs:
            opt.register_output(val_map[out.name])

        return opt

class DeadCodeEliminationPass(Pass):
    def run(self, graph: Graph) -> Graph:
        live_values: Set[str] = {out.name for out in graph.outputs}
        for node in reversed(graph.nodes):
            if any(out.name in live_values for out in node.outputs):
                for inp in node.inputs:
                    live_values.add(inp.name)

        opt = Graph(graph.name)
        val_map: Dict[str, Value] = {}

        for inp in graph.inputs:
            if inp.name in live_values:
                val_map[inp.name] = opt.add_input(inp.name, inp.shape, inp.dtype)
        for c in graph.constants:
            if c.name in live_values:
                val_map[c.name] = opt.add_constant(c.const_data, prefix=c.name.rstrip("0123456789"))

        for node in graph.nodes:
            if any(out.name in live_values for out in node.outputs):
                mapped_inps = [val_map[inp.name] for inp in node.inputs]
                out_v = opt.add_node(node.op_kind, mapped_inps, node.outputs[0].shape)
                val_map[node.outputs[0].name] = out_v

        for out in graph.outputs:
            opt.register_output(val_map[out.name])

        return opt

class CUDACodeGen:
    @staticmethod
    def generate_ptx_for_fused_elementwise(ops: List, kernel_name: str = "custom_fused_kernel") -> str:
        ptx_lines = [
            ".version 7.0",
            ".target sm_50",
            ".address_size 64",
            f".visible .entry {kernel_name}(",
            "    .param .u64 param_in,",
            "    .param .u64 param_out,",
            "    .param .s64 param_N",
            ") {",
            "    .reg .b32 %r<6>;",
            "    .reg .b64 %rd<12>;",
            "    .reg .f32 %f<10>;",
            "    .reg .pred %p<4>;",
            "    ld.param.u64 %rd1, [param_in];",
            "    ld.param.u64 %rd2, [param_out];",
            "    ld.param.s64 %rd3, [param_N];",
            "    mov.u32 %r1, %ctaid.x;",
            "    mov.u32 %r2, %ntid.x;",
            "    mov.u32 %r3, %tid.x;",
            "    mad.lo.u32 %r4, %r1, %r2, %r3;",
            "    cvt.s64.s32 %rd4, %r4;",
            "    setp.ge.s64 %p1, %rd4, %rd3;",
            "    @%p1 bra KERNEL_DONE;",
            "    shl.b64 %rd5, %rd4, 2;",
            "    add.s64 %rd6, %rd1, %rd5;",
            "    ld.global.f32 %f1, [%rd6];"
        ]
        curr_reg = "%f1"
        reg_idx = 2
        for item in ops:
            op = item[0] if isinstance(item, (tuple, list)) else item
            if op == OpKind.RELU:
                ptx_lines.append("    mov.f32 %f" + str(reg_idx) + ", 0f00000000;")
                ptx_lines.append("    max.f32 %f" + str(reg_idx + 1) + ", " + curr_reg + ", %f" + str(reg_idx) + ";")
                curr_reg = "%f" + str(reg_idx + 1)
                reg_idx += 2
        ptx_lines.extend([
            "    add.s64 %rd7, %rd2, %rd5;",
            "    st.global.f32 [%rd7], " + curr_reg + ";",
            "KERNEL_DONE:",
            "    ret;",
            "}"
        ])
        return "\n".join(ptx_lines)


class TracerTensor:
    def __init__(self, value: Value, graph: Graph):
        self.value = value
        self.graph = graph
        self.shape = value.shape

    def _wrap_operand(self, other) -> Value:
        if isinstance(other, TracerTensor):
            return other.value
        elif isinstance(other, (int, float)):
            arr = np.array([other], dtype=np.float32)
            return self.graph.add_constant(arr)
        elif isinstance(other, np.ndarray):
            return self.graph.add_constant(other)
        else:
            return self.graph.new_value(getattr(other, "shape", (1,)))

    def __matmul__(self, other):
        other_val = self._wrap_operand(other)
        out_shape = (self.shape[0], other_val.shape[1]) if len(self.shape) == 2 and len(other_val.shape) == 2 else self.shape
        out_val = self.graph.add_node(OpKind.MATMUL, [self.value, other_val], out_shape)
        return TracerTensor(out_val, self.graph)

    def __add__(self, other):
        other_val = self._wrap_operand(other)
        out_val = self.graph.add_node(OpKind.ADD, [self.value, other_val], self.shape)
        return TracerTensor(out_val, self.graph)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_val = self._wrap_operand(other)
        out_val = self.graph.add_node(OpKind.SUB, [self.value, other_val], self.shape)
        return TracerTensor(out_val, self.graph)

    def __rsub__(self, other):
        other_val = self._wrap_operand(other)
        out_val = self.graph.add_node(OpKind.SUB, [other_val, self.value], self.shape)
        return TracerTensor(out_val, self.graph)

    def __mul__(self, other):
        other_val = self._wrap_operand(other)
        out_val = self.graph.add_node(OpKind.MUL, [self.value, other_val], self.shape)
        return TracerTensor(out_val, self.graph)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other_val = self._wrap_operand(other)
        out_val = self.graph.add_node(OpKind.DIV, [self.value, other_val], self.shape)
        return TracerTensor(out_val, self.graph)

    def __rtruediv__(self, other):
        other_val = self._wrap_operand(other)
        out_val = self.graph.add_node(OpKind.DIV, [other_val, self.value], self.shape)
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

        for c in self.graph.constants:
            env[c.name] = Tensor(c.const_data, device=self.device)

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
            elif node.op_kind == OpKind.FUSED_ADD_RELU:
                out = add_relu(inps[0], inps[1])
            elif node.op_kind == OpKind.FUSED_MATMUL_RELU:
                out = matmul_relu(inps[0], inps[1])
            elif node.op_kind == OpKind.FUSED_MATMUL_ADD:
                out = matmul_add(inps[0], inps[1], inps[2])
            elif node.op_kind == OpKind.FUSED_MATMUL_ADD_RELU:
                out = matmul_add_relu(inps[0], inps[1], inps[2])
            elif node.op_kind == OpKind.FUSED_MATMUL_ADD_RELU_RESIDUAL:
                out = matmul_add_relu_residual(inps[0], inps[1], inps[2], inps[3])
            else:
                raise NotImplementedError(f"Unsupported OpKind: {node.op_kind}")

            env[node.outputs[0].name] = out

        results = [env[v.name] for v in self.graph.outputs]
        return results[0] if len(results) == 1 else tuple(results)

def compile(fn_or_graph, *example_inputs, target: str = "gpu", optimize: bool = True) -> CompiledGraph:
    if isinstance(fn_or_graph, Graph):
        g = fn_or_graph
    elif callable(fn_or_graph):
        g = trace(fn_or_graph, *example_inputs)
    else:
        raise ValueError("Pass callable function with example inputs or a traced Graph.")
    if optimize:
        g = g.optimize()
    return CompiledGraph(g, target=target)


