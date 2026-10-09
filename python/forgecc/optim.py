import ctypes
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .core import Tensor, Device, is_gpu_available, _rt

class Optimizer:
    def __init__(self, params: List[Tensor], lr: float = 0.001, weight_decay: float = 0.0):
        self.params = [p for p in params if isinstance(p, Tensor)]
        self.lr = lr
        self.weight_decay = weight_decay
        self.state: Dict[int, Dict[str, Any]] = {}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.momentum = momentum
        for p in self.params:
            if self.momentum > 0.0:
                self.state[id(p)] = {
                    "velocity": np.zeros(p.shape, dtype=np.float32)
                }

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            n = int(np.prod(p.shape))
            w_arr = p.numpy()
            g_arr = p.grad.numpy()

            vel_ptr = None
            if self.momentum > 0.0:
                vel_arr = self.state[id(p)]["velocity"]
                vel_ptr = vel_arr.ctypes.data_as(ctypes.c_void_p)

            _rt.lib.cpu_sgd_step(
                w_arr.ctypes.data_as(ctypes.c_void_p),
                g_arr.ctypes.data_as(ctypes.c_void_p),
                vel_ptr,
                float(self.lr),
                float(self.momentum),
                float(self.weight_decay),
                n
            )
            p._set_from_numpy(w_arr)

class Adam(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_num = 0
        for p in self.params:
            self.state[id(p)] = {
                "m": np.zeros(p.shape, dtype=np.float32),
                "v": np.zeros(p.shape, dtype=np.float32)
            }

    def step(self):
        self.step_num += 1
        for p in self.params:
            if p.grad is None:
                continue
            n = int(np.prod(p.shape))
            w_arr = p.numpy()
            g_arr = p.grad.numpy()
            m_arr = self.state[id(p)]["m"]
            v_arr = self.state[id(p)]["v"]

            _rt.lib.cpu_adam_step(
                w_arr.ctypes.data_as(ctypes.c_void_p),
                g_arr.ctypes.data_as(ctypes.c_void_p),
                m_arr.ctypes.data_as(ctypes.c_void_p),
                v_arr.ctypes.data_as(ctypes.c_void_p),
                float(self.lr),
                float(self.beta1),
                float(self.beta2),
                float(self.eps),
                float(self.weight_decay),
                int(self.step_num),
                n
            )
            p._set_from_numpy(w_arr)

class AdamW(Adam):
    def __init__(self, params: List[Tensor], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
