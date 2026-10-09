import ctypes
import numpy as np
from typing import List, Tuple, Optional, Any
from .core import Tensor, Device, is_gpu_available, _rt, matmul, add, sub, mul, div, relu, sigmoid, gelu, tanh

class GradFn:
    def __init__(self):
        self.next_functions: List[Tuple[Optional['GradFn'], 'Tensor']] = []

    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        raise NotImplementedError

class MatMulBackward(GradFn):
    def __init__(self, A: Tensor, B: Tensor):
        super().__init__()
        self.A = A
        self.B = B

    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        grad_A = None
        grad_B = None

        M, K = self.A.shape
        _, N = self.B.shape

        arr_grad_out = grad_output.numpy()
        arr_A = self.A.numpy()
        arr_B = self.B.numpy()

        if self.A.requires_grad:
            arr_gA = np.zeros((M, K), dtype=np.float32)
            _rt.lib.cpu_matmul_backward(
                arr_grad_out.ctypes.data_as(ctypes.c_void_p),
                arr_A.ctypes.data_as(ctypes.c_void_p),
                arr_B.ctypes.data_as(ctypes.c_void_p),
                arr_gA.ctypes.data_as(ctypes.c_void_p),
                None,
                M, K, N
            )
            grad_A = Tensor(arr_gA, device=self.A.device)

        if self.B.requires_grad:
            arr_gB = np.zeros((K, N), dtype=np.float32)
            _rt.lib.cpu_matmul_backward(
                arr_grad_out.ctypes.data_as(ctypes.c_void_p),
                arr_A.ctypes.data_as(ctypes.c_void_p),
                arr_B.ctypes.data_as(ctypes.c_void_p),
                None,
                arr_gB.ctypes.data_as(ctypes.c_void_p),
                M, K, N
            )
            grad_B = Tensor(arr_gB, device=self.B.device)

        return [grad_A, grad_B]

def _reduce_grad_shape(grad_arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    if grad_arr.shape == target_shape:
        return grad_arr
    g = grad_arr
    while len(g.shape) > len(target_shape):
        g = np.sum(g, axis=0)
    for i, (gd, td) in enumerate(zip(g.shape, target_shape)):
        if td == 1 and gd > 1:
            g = np.sum(g, axis=i, keepdims=True)
    if g.shape != target_shape:
        g = np.reshape(g, target_shape)
    return g

class AddBackward(GradFn):
    def __init__(self, A: Tensor, B: Tensor):
        super().__init__()
        self.A = A
        self.B = B

    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        g_arr = grad_output.numpy()
        grad_A = Tensor(_reduce_grad_shape(g_arr, self.A.shape), device=self.A.device) if self.A.requires_grad else None
        grad_B = Tensor(_reduce_grad_shape(g_arr, self.B.shape), device=self.B.device) if self.B.requires_grad else None
        return [grad_A, grad_B]

class SubBackward(GradFn):
    def __init__(self, A: Tensor, B: Tensor):
        super().__init__()
        self.A = A
        self.B = B

    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        g_arr = grad_output.numpy()
        grad_A = Tensor(_reduce_grad_shape(g_arr, self.A.shape), device=self.A.device) if self.A.requires_grad else None
        grad_B = Tensor(_reduce_grad_shape(-g_arr, self.B.shape), device=self.B.device) if self.B.requires_grad else None
        return [grad_A, grad_B]

class MulBackward(GradFn):
    def __init__(self, A: Tensor, B: Tensor):
        super().__init__()
        self.A = A
        self.B = B

    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        grad_A = None
        grad_B = None
        g_arr = grad_output.numpy()
        if self.A.requires_grad:
            g_A = _reduce_grad_shape(g_arr * self.B.numpy(), self.A.shape)
            grad_A = Tensor(g_A, device=self.A.device)
        if self.B.requires_grad:
            g_B = _reduce_grad_shape(g_arr * self.A.numpy(), self.B.shape)
            grad_B = Tensor(g_B, device=self.B.device)
        return [grad_A, grad_B]

class DivBackward(GradFn):
    def __init__(self, A: Tensor, B: Tensor):
        super().__init__()
        self.A = A
        self.B = B

    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        grad_A = None
        grad_B = None
        b_arr = self.B.numpy()
        g_arr = grad_output.numpy()
        a_arr = self.A.numpy()
        if self.A.requires_grad:
            g_A = _reduce_grad_shape(g_arr / b_arr, self.A.shape)
            grad_A = Tensor(g_A, device=self.A.device)
        if self.B.requires_grad:
            g_B = _reduce_grad_shape(-g_arr * a_arr / (b_arr * b_arr), self.B.shape)
            grad_B = Tensor(g_B, device=self.B.device)
        return [grad_A, grad_B]

class ReluBackward(GradFn):
    def __init__(self, in_tensor: Tensor):
        super().__init__()
        self.in_tensor = in_tensor

    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        if not self.in_tensor.requires_grad:
            return [None]
        n = int(np.prod(self.in_tensor.shape))
        g_out_arr = grad_output.numpy()
        in_arr = self.in_tensor.numpy()
        g_in_arr = np.empty_like(in_arr)
        _rt.lib.cpu_relu_backward(
            g_out_arr.ctypes.data_as(ctypes.c_void_p),
            in_arr.ctypes.data_as(ctypes.c_void_p),
            g_in_arr.ctypes.data_as(ctypes.c_void_p),
            n
        )
        return [Tensor(g_in_arr, device=self.in_tensor.device)]

class SigmoidBackward(GradFn):
    def __init__(self, out_tensor: Tensor, in_tensor: Tensor):
        super().__init__()
        self.out_tensor = out_tensor
        self.in_tensor = in_tensor

    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        if not self.in_tensor.requires_grad:
            return [None]
        s = self.out_tensor.numpy()
        g_in = grad_output.numpy() * s * (1.0 - s)
        return [Tensor(g_in, device=self.in_tensor.device)]

class TanhBackward(GradFn):
    def __init__(self, out_tensor: Tensor, in_tensor: Tensor):
        super().__init__()
        self.out_tensor = out_tensor
        self.in_tensor = in_tensor

    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        if not self.in_tensor.requires_grad:
            return [None]
        t = self.out_tensor.numpy()
        g_in = grad_output.numpy() * (1.0 - t * t)
        return [Tensor(g_in, device=self.in_tensor.device)]

class MSELossBackward(GradFn):
    def __init__(self, pred: Tensor, target: Tensor):
        super().__init__()
        self.pred = pred
        self.target = target

    def backward(self, grad_output: Tensor) -> List[Optional[Tensor]]:
        n = float(np.prod(self.pred.shape))
        scale = 2.0 / n
        diff = self.pred.numpy() - self.target.numpy()
        g_pred = diff * scale * float(grad_output.numpy().item() if grad_output.shape == (1,) else 1.0)
        grad_pred = Tensor(g_pred, device=self.pred.device) if self.pred.requires_grad else None
        return [grad_pred, None]

def backward_tape(root_tensor: Tensor, grad: Optional[Tensor] = None):
    if grad is None:
        grad = Tensor(np.ones(root_tensor.shape, dtype=np.float32), device=root_tensor.device)

    visited = set()
    topo_order: List[Tensor] = []

    def build_topo(t: Tensor):
        if t not in visited:
            visited.add(t)
            if t.grad_fn:
                for _, parent_t in t.grad_fn.next_functions:
                    if parent_t:
                        build_topo(parent_t)
            topo_order.append(t)

    build_topo(root_tensor)

    grads: dict = {root_tensor: grad}

    for t in reversed(topo_order):
        if t not in grads:
            continue
        g = grads[t]
        if t.grad_fn:
            parent_grads = t.grad_fn.backward(g)
            for (fn, parent_t), p_grad in zip(t.grad_fn.next_functions, parent_grads):
                if parent_t is not None and p_grad is not None:
                    if parent_t not in grads:
                        grads[parent_t] = p_grad
                    else:
                        grads[parent_t] = Tensor(grads[parent_t].numpy() + p_grad.numpy(), device=parent_t.device)

    for t in topo_order:
        if getattr(t, "requires_grad", False) and t in grads:
            if t.grad is None:
                t.grad = grads[t]
            else:
                t.grad = Tensor(t.grad.numpy() + grads[t].numpy(), device=t.device)

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred.numpy() - target.numpy()
    loss_val = np.mean(diff * diff)
    out = Tensor(np.array([loss_val], dtype=np.float32), device=pred.device)
    out.requires_grad = pred.requires_grad or target.requires_grad
    if out.requires_grad:
        fn = MSELossBackward(pred, target)
        fn.next_functions = [(getattr(pred, "grad_fn", None), pred), (getattr(target, "grad_fn", None), target)]
        out.grad_fn = fn
    return out
