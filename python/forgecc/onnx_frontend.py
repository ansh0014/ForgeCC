import struct
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from .ir import Graph, OpKind, Value, CompiledGraph
from .core import Tensor, Device, is_gpu_available

class ProtoReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def has_more(self) -> bool:
        return self.pos < len(self.data)

    def read_varint(self) -> int:
        res = 0
        shift = 0
        while self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            res |= (b & 0x7F) << shift
            if (b & 0x80) == 0:
                break
            shift += 7
        return res

    def read_tag(self) -> Tuple[int, int]:
        if not self.has_more():
            return 0, 0
        v = self.read_varint()
        field_num = v >> 3
        wire_type = v & 0x07
        return field_num, wire_type

    def read_length_delimited(self) -> bytes:
        length = self.read_varint()
        res = self.data[self.pos:self.pos + length]
        self.pos += length
        return res

    def read_string(self) -> str:
        raw = self.read_length_delimited()
        return raw.decode("utf-8", errors="ignore")

    def read_fixed32(self) -> int:
        res = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return res

    def read_fixed64(self) -> int:
        res = struct.unpack_from("<Q", self.data, self.pos)[0]
        self.pos += 8
        return res

    def skip_field(self, wire_type: int):
        if wire_type == 0:
            self.read_varint()
        elif wire_type == 1:
            self.pos += 8
        elif wire_type == 2:
            length = self.read_varint()
            self.pos += length
        elif wire_type == 5:
            self.pos += 4
        else:
            raise ValueError(f"Unknown wire type {wire_type}")

class ONNXTensor:
    def __init__(self):
        self.name = ""
        self.dims: List[int] = []
        self.data_type = 1
        self.raw_data = b""
        self.float_data: List[float] = []

    def to_numpy(self) -> np.ndarray:
        shape = tuple(self.dims) if self.dims else (1,)
        if self.raw_data:
            arr = np.frombuffer(self.raw_data, dtype=np.float32).reshape(shape)
            return arr.copy()
        elif self.float_data:
            return np.array(self.float_data, dtype=np.float32).reshape(shape)
        else:
            return np.zeros(shape, dtype=np.float32)

class ONNXNode:
    def __init__(self):
        self.name = ""
        self.op_type = ""
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.attributes: Dict[str, Any] = {}

class ONNXValueInfo:
    def __init__(self):
        self.name = ""
        self.dims: List[int] = []

class ONNXGraph:
    def __init__(self):
        self.name = ""
        self.nodes: List[ONNXNode] = []
        self.initializers: Dict[str, ONNXTensor] = {}
        self.inputs: List[ONNXValueInfo] = []
        self.outputs: List[ONNXValueInfo] = []

def _parse_tensor_proto(data: bytes) -> ONNXTensor:
    t = ONNXTensor()
    r = ProtoReader(data)
    while r.has_more():
        field_num, wire_type = r.read_tag()
        if field_num == 1:
            if wire_type == 2:
                sub_r = ProtoReader(r.read_length_delimited())
                while sub_r.has_more():
                    t.dims.append(sub_r.read_varint())
            else:
                t.dims.append(r.read_varint())
        elif field_num == 2:
            t.data_type = r.read_varint()
        elif field_num == 4:
            if wire_type == 2:
                raw = r.read_length_delimited()
                count = len(raw) // 4
                t.float_data.extend(struct.unpack(f"<{count}f", raw))
            elif wire_type == 5:
                val = struct.unpack("<f", struct.pack("<I", r.read_fixed32()))[0]
                t.float_data.append(val)
        elif field_num == 7:
            t.name = r.read_string()
        elif field_num == 9:
            t.raw_data = r.read_length_delimited()
        else:
            r.skip_field(wire_type)
    return t

def _parse_node_proto(data: bytes) -> ONNXNode:
    node = ONNXNode()
    r = ProtoReader(data)
    while r.has_more():
        field_num, wire_type = r.read_tag()
        if field_num == 1:
            node.inputs.append(r.read_string())
        elif field_num == 2:
            node.outputs.append(r.read_string())
        elif field_num == 3:
            node.name = r.read_string()
        elif field_num == 4:
            node.op_type = r.read_string()
        else:
            r.skip_field(wire_type)
    return node

def _parse_value_info_proto(data: bytes) -> ONNXValueInfo:
    vi = ONNXValueInfo()
    r = ProtoReader(data)
    while r.has_more():
        field_num, wire_type = r.read_tag()
        if field_num == 1:
            vi.name = r.read_string()
        elif field_num == 2:
            type_raw = r.read_length_delimited()
            type_r = ProtoReader(type_raw)
            while type_r.has_more():
                tf_num, tw_type = type_r.read_tag()
                if tf_num == 1:
                    tensor_raw = type_r.read_length_delimited()
                    ten_r = ProtoReader(tensor_raw)
                    while ten_r.has_more():
                        sf_num, sw_type = ten_r.read_tag()
                        if sf_num == 1:
                            shape_raw = ten_r.read_length_delimited()
                            sh_r = ProtoReader(shape_raw)
                            while sh_r.has_more():
                                df_num, dw_type = sh_r.read_tag()
                                if df_num == 1:
                                    dim_raw = sh_r.read_length_delimited()
                                    dim_r = ProtoReader(dim_raw)
                                    while dim_r.has_more():
                                        f_dim, w_dim = dim_r.read_tag()
                                        if f_dim == 1:
                                            vi.dims.append(dim_r.read_varint())
                                        else:
                                            dim_r.skip_field(w_dim)
                                else:
                                    sh_r.skip_field(dw_type)
                        else:
                            ten_r.skip_field(sw_type)
                else:
                    type_r.skip_field(tw_type)
        else:
            r.skip_field(wire_type)
    return vi

def _parse_graph_proto(data: bytes) -> ONNXGraph:
    graph = ONNXGraph()
    r = ProtoReader(data)
    while r.has_more():
        field_num, wire_type = r.read_tag()
        if field_num == 1:
            graph.nodes.append(_parse_node_proto(r.read_length_delimited()))
        elif field_num == 2:
            graph.name = r.read_string()
        elif field_num == 5:
            tensor = _parse_tensor_proto(r.read_length_delimited())
            graph.initializers[tensor.name] = tensor
        elif field_num == 11:
            graph.inputs.append(_parse_value_info_proto(r.read_length_delimited()))
        elif field_num == 12:
            graph.outputs.append(_parse_value_info_proto(r.read_length_delimited()))
        else:
            r.skip_field(wire_type)
    return graph

def _parse_model_proto(data: bytes) -> ONNXGraph:
    r = ProtoReader(data)
    while r.has_more():
        field_num, wire_type = r.read_tag()
        if field_num == 7:
            return _parse_graph_proto(r.read_length_delimited())
        else:
            r.skip_field(wire_type)
    return ONNXGraph()

def from_onnx(model_or_bytes: Union[str, bytes, bytearray, dict]) -> Graph:
    if isinstance(model_or_bytes, dict):
        return _from_dict(model_or_bytes)

    if isinstance(model_or_bytes, str):
        with open(model_or_bytes, "rb") as f:
            raw_bytes = f.read()
    else:
        raw_bytes = bytes(model_or_bytes)

    onnx_graph = _parse_model_proto(raw_bytes)
    ir_graph = Graph(onnx_graph.name if onnx_graph.name else "onnx_imported_graph")

    value_env: Dict[str, Value] = {}

    for name, tensor in onnx_graph.initializers.items():
        arr = tensor.to_numpy()
        value_env[name] = ir_graph.add_constant(arr, name=name)

    for inp in onnx_graph.inputs:
        if inp.name in value_env:
            continue
        shape = tuple(inp.dims) if inp.dims else (1,)
        value_env[inp.name] = ir_graph.add_input(inp.name, shape)

    op_mapping = {
        "MatMul": OpKind.MATMUL,
        "Gemm": OpKind.MATMUL,
        "Relu": OpKind.RELU,
        "Add": OpKind.ADD,
        "Sub": OpKind.SUB,
        "Mul": OpKind.MUL,
        "Div": OpKind.DIV,
        "Sigmoid": OpKind.SIGMOID,
        "Tanh": OpKind.TANH,
        "ReduceSum": OpKind.SUM,
        "ReduceMean": OpKind.MEAN
    }

    for node in onnx_graph.nodes:
        op_kind = op_mapping.get(node.op_type)
        if op_kind is None:
            raise NotImplementedError(f"Unsupported ONNX operator: {node.op_type}")

        input_values = []
        for inp_name in node.inputs:
            if inp_name not in value_env:
                value_env[inp_name] = ir_graph.add_input(inp_name, (1,))
            input_values.append(value_env[inp_name])

        out_shape = (1,)
        if op_kind == OpKind.MATMUL and len(input_values) >= 2:
            s0 = input_values[0].shape
            s1 = input_values[1].shape
            if len(s0) == 2 and len(s1) == 2:
                out_shape = (s0[0], s1[1])
        elif input_values:
            out_shape = input_values[0].shape

        out_val = ir_graph.add_node(op_kind, input_values, out_shape)
        if node.outputs:
            value_env[node.outputs[0]] = out_val

    for out in onnx_graph.outputs:
        if out.name in value_env:
            ir_graph.register_output(value_env[out.name])

    return ir_graph

def _from_dict(model_dict: dict) -> Graph:
    ir_graph = Graph(model_dict.get("name", "onnx_model"))
    value_env: Dict[str, Value] = {}

    for name, data in model_dict.get("initializers", {}).items():
        arr = np.array(data, dtype=np.float32)
        value_env[name] = ir_graph.add_constant(arr, name=name)

    for name, shape in model_dict.get("inputs", {}).items():
        if name not in value_env:
            value_env[name] = ir_graph.add_input(name, tuple(shape))

    op_mapping = {
        "MatMul": OpKind.MATMUL,
        "Gemm": OpKind.MATMUL,
        "Relu": OpKind.RELU,
        "Add": OpKind.ADD,
        "Sub": OpKind.SUB,
        "Mul": OpKind.MUL,
        "Div": OpKind.DIV,
        "Sigmoid": OpKind.SIGMOID,
        "Tanh": OpKind.TANH,
        "ReduceSum": OpKind.SUM,
        "ReduceMean": OpKind.MEAN
    }

    for node_info in model_dict.get("nodes", []):
        op_type = node_info["op_type"]
        inputs = node_info["inputs"]
        outputs = node_info["outputs"]

        op_kind = op_mapping.get(op_type)
        if op_kind is None:
            raise NotImplementedError(f"Unsupported ONNX operator: {op_type}")

        input_values = [value_env[inp_name] for inp_name in inputs]
        out_shape = (1,)
        if op_kind == OpKind.MATMUL and len(input_values) >= 2:
            s0 = input_values[0].shape
            s1 = input_values[1].shape
            if len(s0) == 2 and len(s1) == 2:
                out_shape = (s0[0], s1[1])
        elif input_values:
            out_shape = input_values[0].shape

        out_val = ir_graph.add_node(op_kind, input_values, out_shape)
        if outputs:
            value_env[outputs[0]] = out_val

    for out_name in model_dict.get("outputs", []):
        if out_name in value_env:
            ir_graph.register_output(value_env[out_name])

    return ir_graph

def compile_onnx(model_or_bytes: Union[str, bytes, bytearray, dict], target: str = "gpu", optimize: bool = True) -> CompiledGraph:
    g = from_onnx(model_or_bytes)
    if optimize:
        g = g.optimize()
    return CompiledGraph(g, target=target)
