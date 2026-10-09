import os
import sys
import ctypes
import numpy as np
from pathlib import Path
from typing import Dict, List

def _setup_dll_search_path():
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        mingw_bin = "C:/msys64/ucrt64/bin"
        if os.path.isdir(mingw_bin):
            try:
                os.add_dll_directory(mingw_bin)
            except Exception:
                pass
        pkg_dir = Path(__file__).parent.resolve()
        try:
            os.add_dll_directory(str(pkg_dir))
        except Exception:
            pass

_setup_dll_search_path()

def _find_library():
    current_dir = Path(__file__).parent.resolve()
    candidates = [
        current_dir / "libforgecc_rt.dll",
        current_dir / "libforgecc_rt.so",
        current_dir / "libforgecc_rt.dylib",
        current_dir.parent.parent / "build" / "runtime" / "libforgecc_rt.dll",
        current_dir.parent.parent / "build" / "runtime" / "libforgecc_rt.so",
        current_dir.parent.parent / "build" / "runtime" / "Release" / "forgecc_rt.dll"
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

class Device:
    CPU = 0
    GPU = 1

class GPUMemoryPool:
    def __init__(self):
        self._free_blocks: Dict[int, List[int]] = {}
        self._allocated_count = 0
        self._cached_count = 0

    def allocate(self, bytes_size: int, rt_lib) -> int:
        bucket = 1 << (bytes_size - 1).bit_length() if bytes_size > 0 else 256
        bucket = max(bucket, 256)
        if bucket in self._free_blocks and self._free_blocks[bucket]:
            ptr = self._free_blocks[bucket].pop()
            self._cached_count -= 1
            return ptr
        ptr = rt_lib.forge_gpu_malloc(bucket)
        if ptr:
            self._allocated_count += 1
        return ptr

    def free(self, ptr: int, bytes_size: int):
        if not ptr:
            return
        bucket = 1 << (bytes_size - 1).bit_length() if bytes_size > 0 else 256
        bucket = max(bucket, 256)
        if bucket not in self._free_blocks:
            self._free_blocks[bucket] = []
        self._free_blocks[bucket].append(ptr)
        self._cached_count += 1

    def stats(self) -> Dict[str, int]:
        return {
            "total_allocations": self._allocated_count,
            "cached_buffers": self._cached_count
        }

    def clear(self, rt_lib):
        for bucket, ptrs in self._free_blocks.items():
            for p in ptrs:
                rt_lib.forge_gpu_free(p)
        self._free_blocks.clear()
        self._cached_count = 0

class ForgeRuntime:
    def __init__(self):
        self.lib = None
        self.mempool = GPUMemoryPool()
        _setup_dll_search_path()
        lib_path = _find_library()
        if not lib_path:
            return
        
        try:
            self.lib = ctypes.CDLL(lib_path)
            self._setup_signatures()
        except Exception:
            self.lib = None

    def _setup_signatures(self):
        self.lib.forge_gpu_is_available.restype = ctypes.c_bool
        self.lib.forge_gpu_is_available.argtypes = []

        self.lib.forge_gpu_sync.restype = None
        self.lib.forge_gpu_sync.argtypes = []

        self.lib.forge_gpu_malloc.restype = ctypes.c_void_p
        self.lib.forge_gpu_malloc.argtypes = [ctypes.c_size_t]

        self.lib.forge_gpu_free.restype = None
        self.lib.forge_gpu_free.argtypes = [ctypes.c_void_p]

        self.lib.forge_gpu_memcpy_to_device.restype = None
        self.lib.forge_gpu_memcpy_to_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

        self.lib.forge_gpu_memcpy_to_host.restype = None
        self.lib.forge_gpu_memcpy_to_host.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

        self.lib.forge_gpu_matmul.restype = None
        self.lib.forge_gpu_matmul.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        self.lib.forge_gpu_relu.restype = None
        self.lib.forge_gpu_relu.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.forge_gpu_add.restype = None
        self.lib.forge_gpu_add.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.forge_gpu_add_relu_fused.restype = None
        self.lib.forge_gpu_add_relu_fused.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.forge_gpu_sub.restype = None
        self.lib.forge_gpu_sub.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.forge_gpu_mul.restype = None
        self.lib.forge_gpu_mul.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.forge_gpu_matmul_relu_fused.restype = None
        self.lib.forge_gpu_matmul_relu_fused.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        self.lib.forge_gpu_matmul_add_fused.restype = None
        self.lib.forge_gpu_matmul_add_fused.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        self.lib.forge_gpu_matmul_add_relu_fused.restype = None
        self.lib.forge_gpu_matmul_add_relu_fused.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        self.lib.forge_gpu_matmul_add_relu_residual_fused.restype = None
        self.lib.forge_gpu_matmul_add_relu_residual_fused.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        self.lib.cpu_matmul.restype = None
        self.lib.cpu_matmul.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        self.lib.cpu_relu.restype = None
        self.lib.cpu_relu.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.cpu_sigmoid.restype = None
        self.lib.cpu_sigmoid.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.cpu_tanh_act.restype = None
        self.lib.cpu_tanh_act.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.cpu_gelu.restype = None
        self.lib.cpu_gelu.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.cpu_add.restype = None
        self.lib.cpu_add.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.cpu_add_relu.restype = None
        self.lib.cpu_add_relu.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.cpu_sub.restype = None
        self.lib.cpu_sub.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.cpu_mul.restype = None
        self.lib.cpu_mul.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.cpu_div.restype = None
        self.lib.cpu_div.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.cpu_sum.restype = None
        self.lib.cpu_sum.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.cpu_mean.restype = None
        self.lib.cpu_mean.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.cpu_matmul_add.restype = None
        self.lib.cpu_matmul_add.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        self.lib.cpu_matmul_add_relu.restype = None
        self.lib.cpu_matmul_add_relu.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        self.lib.cpu_matmul_add_relu_residual.restype = None
        self.lib.cpu_matmul_add_relu_residual.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        self.lib.forge_gpu_get_device_name.restype = ctypes.c_bool
        self.lib.forge_gpu_get_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]

        self.lib.forge_gpu_get_attribute.restype = ctypes.c_int
        self.lib.forge_gpu_get_attribute.argtypes = [ctypes.c_int]

        self.lib.forge_gpu_get_memory_info.restype = ctypes.c_bool
        self.lib.forge_gpu_get_memory_info.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]

        self.lib.forge_gpu_auto_tune_elementwise.restype = None
        self.lib.forge_gpu_auto_tune_elementwise.argtypes = [
            ctypes.c_int64, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)
        ]

        self.lib.forge_gpu_auto_tune_matmul.restype = None
        self.lib.forge_gpu_auto_tune_matmul.argtypes = [
            ctypes.c_int64, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)
        ]

        self.lib.cpu_matmul_backward.restype = None
        self.lib.cpu_matmul_backward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
        ]

        self.lib.cpu_relu_backward.restype = None
        self.lib.cpu_relu_backward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.cpu_sgd_step.restype = None
        self.lib.cpu_sgd_step.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int64
        ]

        self.lib.cpu_adam_step.restype = None
        self.lib.cpu_adam_step.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_int64, ctypes.c_int64
        ]

        self.lib.forge_gpu_relu_backward.restype = None
        self.lib.forge_gpu_relu_backward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]

        self.lib.forge_gpu_sgd_step.restype = None
        self.lib.forge_gpu_sgd_step.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int64
        ]

        self.lib.forge_gpu_adam_step.restype = None
        self.lib.forge_gpu_adam_step.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_int64, ctypes.c_int64
        ]

_rt = ForgeRuntime()

def is_gpu_available() -> bool:
    if _rt.lib:
        try:
            return bool(_rt.lib.forge_gpu_is_available())
        except Exception:
            return False
    return False

def sync():
    if _rt.lib:
        _rt.lib.forge_gpu_sync()

def memory_stats() -> Dict[str, int]:
    return _rt.mempool.stats()

def memory_clear():
    if _rt.lib:
        _rt.mempool.clear(_rt.lib)

def get_device_properties(device_id: int = 0) -> Dict:
    if not is_gpu_available():
        return {
            "device": "CPU",
            "available": False
        }
    buf = ctypes.create_string_buffer(256)
    name = "Unknown NVIDIA GPU"
    if _rt.lib.forge_gpu_get_device_name(buf, 256):
        name = buf.value.decode("utf-8")

    sm_count = _rt.lib.forge_gpu_get_attribute(16)
    max_threads = _rt.lib.forge_gpu_get_attribute(1)
    shared_mem = _rt.lib.forge_gpu_get_attribute(8)
    warp_size = _rt.lib.forge_gpu_get_attribute(10)
    major = _rt.lib.forge_gpu_get_attribute(75)
    minor = _rt.lib.forge_gpu_get_attribute(76)

    free_mem = ctypes.c_size_t(0)
    total_mem = ctypes.c_size_t(0)
    _rt.lib.forge_gpu_get_memory_info(ctypes.byref(free_mem), ctypes.byref(total_mem))

    return {
        "device": "GPU",
        "available": True,
        "name": name,
        "sm_count": sm_count if sm_count > 0 else 16,
        "max_threads_per_block": max_threads if max_threads > 0 else 1024,
        "shared_memory_per_block": shared_mem if shared_mem > 0 else 49152,
        "warp_size": warp_size if warp_size > 0 else 32,
        "compute_capability": (major, minor) if major > 0 else (7, 5),
        "free_memory_bytes": free_mem.value,
        "total_memory_bytes": total_mem.value
    }

def auto_tune_elementwise(n: int) -> Dict[str, int]:
    block_size = ctypes.c_uint(256)
    grid_size = ctypes.c_uint(1)
    if is_gpu_available():
        _rt.lib.forge_gpu_auto_tune_elementwise(n, ctypes.byref(block_size), ctypes.byref(grid_size))
    else:
        block_size.value = 256
        grid_size.value = (n + 255) // 256 if n > 0 else 1
    return {
        "block_size": block_size.value,
        "grid_size": grid_size.value
    }

def auto_tune_matmul(M: int, K: int, N: int) -> Dict[str, int]:
    block_x = ctypes.c_uint(16)
    block_y = ctypes.c_uint(16)
    grid_x = ctypes.c_uint(1)
    grid_y = ctypes.c_uint(1)
    if is_gpu_available():
        _rt.lib.forge_gpu_auto_tune_matmul(M, N, ctypes.byref(block_x), ctypes.byref(block_y), ctypes.byref(grid_x), ctypes.byref(grid_y))
    else:
        block_x.value = 16
        block_y.value = 16
        grid_x.value = (N + 15) // 16 if N > 0 else 1
        grid_y.value = (M + 15) // 16 if M > 0 else 1
    return {
        "block_x": block_x.value,
        "block_y": block_y.value,
        "grid_x": grid_x.value,
        "grid_y": grid_y.value
    }


class Tensor:
    def __init__(self, data, device=Device.CPU, dtype=None, requires_grad: bool = False):
        self.gpu_ptr = None
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        target_dtype = dtype if dtype is not None else np.float32
        if isinstance(data, np.ndarray):
            self.host_array = np.ascontiguousarray(data, dtype=target_dtype)
        elif isinstance(data, (list, tuple, float, int)):
            self.host_array = np.ascontiguousarray(np.array(data, dtype=target_dtype))
        elif isinstance(data, Tensor):
            self.host_array = data.numpy().astype(target_dtype).copy()
        else:
            raise TypeError("Unsupported data type for Tensor initialization.")

        self.shape = self.host_array.shape
        self.device = device
        self.nbytes = self.host_array.nbytes

        if self.device == Device.GPU:
            if not is_gpu_available():
                self.device = Device.CPU
            else:
                self.gpu_ptr = _rt.mempool.allocate(self.nbytes, _rt.lib)
                if self.gpu_ptr:
                    _rt.lib.forge_gpu_memcpy_to_device(
                        self.gpu_ptr,
                        self.host_array.ctypes.data_as(ctypes.c_void_p),
                        self.nbytes
                    )
                else:
                    self.device = Device.CPU

    def __del__(self):
        ptr = getattr(self, "gpu_ptr", None)
        if ptr and _rt.lib:
            try:
                _rt.mempool.free(ptr, getattr(self, "nbytes", 0))
            except Exception:
                pass
            self.gpu_ptr = None

    def to(self, target_device):
        if self.device == target_device:
            return self

        if target_device == Device.GPU and is_gpu_available():
            return Tensor(self.numpy(), device=Device.GPU, requires_grad=self.requires_grad)
        else:
            return Tensor(self.numpy(), device=Device.CPU, requires_grad=self.requires_grad)

    def numpy(self) -> np.ndarray:
        if self.device == Device.GPU and self.gpu_ptr and _rt.lib:
            out = np.empty(self.shape, dtype=np.float32)
            _rt.lib.forge_gpu_memcpy_to_host(
                out.ctypes.data_as(ctypes.c_void_p),
                self.gpu_ptr,
                self.nbytes
            )
            return out
        return self.host_array.copy()

    def _set_from_numpy(self, arr: np.ndarray):
        target_dtype = self.host_array.dtype
        self.host_array = np.ascontiguousarray(arr, dtype=target_dtype)
        self.shape = self.host_array.shape
        self.nbytes = self.host_array.nbytes
        if self.device == Device.GPU and self.gpu_ptr and _rt.lib:
            _rt.lib.forge_gpu_memcpy_to_device(
                self.gpu_ptr,
                self.host_array.ctypes.data_as(ctypes.c_void_p),
                self.nbytes
            )

    def backward(self, grad=None):
        from .autograd import backward_tape
        backward_tape(self, grad)

    def __matmul__(self, other):
        return matmul(self, other)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __neg__(self):
        return mul(self, -1.0)

    def relu(self):
        return relu(self)

    def sigmoid(self):
        return sigmoid(self)

    def gelu(self):
        return gelu(self)

    def tanh(self):
        return tanh(self)

    def sum(self):
        return sum_tensor(self)

    def mean(self):
        return mean_tensor(self)

    def t(self):
        arr = np.ascontiguousarray(self.numpy().T)
        return Tensor(arr, device=self.device)

    def reshape(self, *shape):
        arr = np.ascontiguousarray(self.numpy().reshape(*shape))
        return Tensor(arr, device=self.device)

    def __repr__(self):
        dev_str = "GPU" if self.device == Device.GPU else "CPU"
        return f"forgecc.Tensor(shape={list(self.shape)}, device={dev_str}, data=\n{self.numpy()})"

def tensor(data, device=Device.CPU, dtype=None, requires_grad: bool = False) -> Tensor:
    return Tensor(data, device=device, dtype=dtype, requires_grad=requires_grad)


def _to_tensor(x, device=Device.CPU):
    if isinstance(x, Tensor):
        return x
    return Tensor(x, device=device)

def matmul(A, B):
    if hasattr(A, "graph") or hasattr(B, "graph"):
        return A @ B
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)

    if len(t_A.shape) != 2 or len(t_B.shape) != 2:
        raise ValueError("Inputs must be 2D matrices for matmul.")
    if t_A.shape[1] != t_B.shape[0]:
        raise ValueError(f"Shape mismatch: {t_A.shape} and {t_B.shape}")

    M, K = t_A.shape
    _, N = t_B.shape

    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if use_gpu:
        g_A = t_A.to(Device.GPU)
        g_B = t_B.to(Device.GPU)
        C_ptr = _rt.mempool.allocate(M * N * 4, _rt.lib)
        _rt.lib.forge_gpu_matmul(g_A.gpu_ptr, g_B.gpu_ptr, C_ptr, M, K, N)
        _rt.lib.forge_gpu_sync()
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, M * N * 4)
        _rt.mempool.free(C_ptr, M * N * 4)
        out = Tensor(out_arr, device=Device.GPU)
    else:
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.cpu_matmul(
            t_A.numpy().ctypes.data_as(ctypes.c_void_p),
            t_B.numpy().ctypes.data_as(ctypes.c_void_p),
            out_arr.ctypes.data_as(ctypes.c_void_p),
            M, K, N
        )
        out = Tensor(out_arr, device=Device.CPU)

    if t_A.requires_grad or t_B.requires_grad:
        from .autograd import MatMulBackward
        out.requires_grad = True
        fn = MatMulBackward(t_A, t_B)
        fn.next_functions = [(getattr(t_A, "grad_fn", None), t_A), (getattr(t_B, "grad_fn", None), t_B)]
        out.grad_fn = fn
    return out

def add(A, B):
    if hasattr(A, "graph") or hasattr(B, "graph"):
        return A + B
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)

    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if t_A.shape == t_B.shape:
        n = int(np.prod(t_A.shape))
        if use_gpu:
            g_A = t_A.to(Device.GPU)
            g_B = t_B.to(Device.GPU)
            C_ptr = _rt.mempool.allocate(n * 4, _rt.lib)
            _rt.lib.forge_gpu_add(g_A.gpu_ptr, g_B.gpu_ptr, C_ptr, n)
            _rt.lib.forge_gpu_sync()
            out_arr = np.empty(t_A.shape, dtype=np.float32)
            _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, n * 4)
            _rt.mempool.free(C_ptr, n * 4)
            out = Tensor(out_arr, device=Device.GPU)
        else:
            out_arr = np.empty(t_A.shape, dtype=np.float32)
            _rt.lib.cpu_add(
                t_A.numpy().ctypes.data_as(ctypes.c_void_p),
                t_B.numpy().ctypes.data_as(ctypes.c_void_p),
                out_arr.ctypes.data_as(ctypes.c_void_p),
                n
            )
            out = Tensor(out_arr, device=Device.CPU)
    else:
        res = t_A.numpy() + t_B.numpy()
        out = Tensor(res, device=Device.GPU if use_gpu else Device.CPU)

    if t_A.requires_grad or t_B.requires_grad:
        from .autograd import AddBackward
        out.requires_grad = True
        fn = AddBackward(t_A, t_B)
        fn.next_functions = [(getattr(t_A, "grad_fn", None), t_A), (getattr(t_B, "grad_fn", None), t_B)]
        out.grad_fn = fn
    return out

def sub(A, B):
    if hasattr(A, "graph") or hasattr(B, "graph"):
        return A - B
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)
    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if t_A.shape == t_B.shape:
        n = int(np.prod(t_A.shape))
        if use_gpu:
            g_A = t_A.to(Device.GPU)
            g_B = t_B.to(Device.GPU)
            C_ptr = _rt.mempool.allocate(n * 4, _rt.lib)
            _rt.lib.forge_gpu_sub(g_A.gpu_ptr, g_B.gpu_ptr, C_ptr, n)
            _rt.lib.forge_gpu_sync()
            out_arr = np.empty(t_A.shape, dtype=np.float32)
            _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, n * 4)
            _rt.mempool.free(C_ptr, n * 4)
            out = Tensor(out_arr, device=Device.GPU)
        else:
            out_arr = np.empty(t_A.shape, dtype=np.float32)
            _rt.lib.cpu_sub(
                t_A.numpy().ctypes.data_as(ctypes.c_void_p),
                t_B.numpy().ctypes.data_as(ctypes.c_void_p),
                out_arr.ctypes.data_as(ctypes.c_void_p),
                n
            )
            out = Tensor(out_arr, device=Device.CPU)
    else:
        res = t_A.numpy() - t_B.numpy()
        out = Tensor(res, device=Device.GPU if use_gpu else Device.CPU)

    if t_A.requires_grad or t_B.requires_grad:
        from .autograd import SubBackward
        out.requires_grad = True
        fn = SubBackward(t_A, t_B)
        fn.next_functions = [(getattr(t_A, "grad_fn", None), t_A), (getattr(t_B, "grad_fn", None), t_B)]
        out.grad_fn = fn
    return out

def mul(A, B):
    if hasattr(A, "graph") or hasattr(B, "graph"):
        return A * B
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)
    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if t_A.shape == t_B.shape:
        n = int(np.prod(t_A.shape))
        if use_gpu:
            g_A = t_A.to(Device.GPU)
            g_B = t_B.to(Device.GPU)
            C_ptr = _rt.mempool.allocate(n * 4, _rt.lib)
            _rt.lib.forge_gpu_mul(g_A.gpu_ptr, g_B.gpu_ptr, C_ptr, n)
            _rt.lib.forge_gpu_sync()
            out_arr = np.empty(t_A.shape, dtype=np.float32)
            _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, n * 4)
            _rt.mempool.free(C_ptr, n * 4)
            out = Tensor(out_arr, device=Device.GPU)
        else:
            out_arr = np.empty(t_A.shape, dtype=np.float32)
            _rt.lib.cpu_mul(
                t_A.numpy().ctypes.data_as(ctypes.c_void_p),
                t_B.numpy().ctypes.data_as(ctypes.c_void_p),
                out_arr.ctypes.data_as(ctypes.c_void_p),
                n
            )
            out = Tensor(out_arr, device=Device.CPU)
    else:
        res = t_A.numpy() * t_B.numpy()
        out = Tensor(res, device=Device.GPU if use_gpu else Device.CPU)

    if t_A.requires_grad or t_B.requires_grad:
        from .autograd import MulBackward
        out.requires_grad = True
        fn = MulBackward(t_A, t_B)
        fn.next_functions = [(getattr(t_A, "grad_fn", None), t_A), (getattr(t_B, "grad_fn", None), t_B)]
        out.grad_fn = fn
    return out

def div(A, B):
    if hasattr(A, "graph") or hasattr(B, "graph"):
        return A / B
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)
    res = np.divide(t_A.numpy(), t_B.numpy(), out=np.zeros_like(t_A.numpy()), where=(t_B.numpy() != 0))
    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    out = Tensor(res, device=Device.GPU if use_gpu else Device.CPU)
    if t_A.requires_grad or t_B.requires_grad:
        from .autograd import DivBackward
        out.requires_grad = True
        fn = DivBackward(t_A, t_B)
        fn.next_functions = [(getattr(t_A, "grad_fn", None), t_A), (getattr(t_B, "grad_fn", None), t_B)]
        out.grad_fn = fn
    return out

def relu(x):
    if hasattr(x, "graph"):
        return x.relu()
    t_x = _to_tensor(x)
    n = int(np.prod(t_x.shape))
    if t_x.device == Device.GPU and is_gpu_available():
        C_ptr = _rt.mempool.allocate(n * 4, _rt.lib)
        _rt.lib.forge_gpu_relu(t_x.gpu_ptr, C_ptr, n)
        _rt.lib.forge_gpu_sync()
        out_arr = np.empty(t_x.shape, dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, n * 4)
        _rt.mempool.free(C_ptr, n * 4)
        out = Tensor(out_arr, device=Device.GPU)
    else:
        out_arr = np.empty(t_x.shape, dtype=np.float32)
        _rt.lib.cpu_relu(
            t_x.numpy().ctypes.data_as(ctypes.c_void_p),
            out_arr.ctypes.data_as(ctypes.c_void_p),
            n
        )
        out = Tensor(out_arr, device=Device.CPU)

    if t_x.requires_grad:
        from .autograd import ReluBackward
        out.requires_grad = True
        fn = ReluBackward(t_x)
        fn.next_functions = [(getattr(t_x, "grad_fn", None), t_x)]
        out.grad_fn = fn
    return out

def sigmoid(x):
    if hasattr(x, "graph"):
        return x.sigmoid()
    t_x = _to_tensor(x)
    n = int(np.prod(t_x.shape))
    out_arr = np.empty(t_x.shape, dtype=np.float32)
    _rt.lib.cpu_sigmoid(
        t_x.numpy().ctypes.data_as(ctypes.c_void_p),
        out_arr.ctypes.data_as(ctypes.c_void_p),
        n
    )
    out = Tensor(out_arr, device=t_x.device)
    if t_x.requires_grad:
        from .autograd import SigmoidBackward
        out.requires_grad = True
        fn = SigmoidBackward(out, t_x)
        fn.next_functions = [(getattr(t_x, "grad_fn", None), t_x)]
        out.grad_fn = fn
    return out

def gelu(x):
    if hasattr(x, "graph"):
        return x.gelu()
    t_x = _to_tensor(x)
    n = int(np.prod(t_x.shape))
    out_arr = np.empty(t_x.shape, dtype=np.float32)
    _rt.lib.cpu_gelu(
        t_x.numpy().ctypes.data_as(ctypes.c_void_p),
        out_arr.ctypes.data_as(ctypes.c_void_p),
        n
    )
    return Tensor(out_arr, device=t_x.device)

def tanh(x):
    if hasattr(x, "graph"):
        return x.tanh()
    t_x = _to_tensor(x)
    n = int(np.prod(t_x.shape))
    out_arr = np.empty(t_x.shape, dtype=np.float32)
    _rt.lib.cpu_tanh_act(
        t_x.numpy().ctypes.data_as(ctypes.c_void_p),
        out_arr.ctypes.data_as(ctypes.c_void_p),
        n
    )
    out = Tensor(out_arr, device=t_x.device)
    if t_x.requires_grad:
        from .autograd import TanhBackward
        out.requires_grad = True
        fn = TanhBackward(out, t_x)
        fn.next_functions = [(getattr(t_x, "grad_fn", None), t_x)]
        out.grad_fn = fn
    return out

def sum_tensor(x):
    if hasattr(x, "graph"):
        return x.sum()
    t_x = _to_tensor(x)
    n = int(np.prod(t_x.shape))
    res = np.empty((1,), dtype=np.float32)
    _rt.lib.cpu_sum(
        t_x.numpy().ctypes.data_as(ctypes.c_void_p),
        res.ctypes.data_as(ctypes.c_void_p),
        n
    )
    return Tensor(res, device=t_x.device)

def mean_tensor(x):
    if hasattr(x, "graph"):
        return x.mean()
    t_x = _to_tensor(x)
    n = int(np.prod(t_x.shape))
    res = np.empty((1,), dtype=np.float32)
    _rt.lib.cpu_mean(
        t_x.numpy().ctypes.data_as(ctypes.c_void_p),
        res.ctypes.data_as(ctypes.c_void_p),
        n
    )
    return Tensor(res, device=t_x.device)

def matmul_relu(A, B):
    if hasattr(A, "graph") or hasattr(B, "graph"):
        return (A @ B).relu()
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)
    M, K = t_A.shape
    _, N = t_B.shape
    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if use_gpu:
        g_A = t_A.to(Device.GPU)
        g_B = t_B.to(Device.GPU)
        C_ptr = _rt.mempool.allocate(M * N * 4, _rt.lib)
        _rt.lib.forge_gpu_matmul_relu_fused(g_A.gpu_ptr, g_B.gpu_ptr, C_ptr, M, K, N)
        _rt.lib.forge_gpu_sync()
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, M * N * 4)
        _rt.mempool.free(C_ptr, M * N * 4)
        return Tensor(out_arr, device=Device.GPU)
    else:
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.cpu_matmul(
            t_A.numpy().ctypes.data_as(ctypes.c_void_p),
            t_B.numpy().ctypes.data_as(ctypes.c_void_p),
            out_arr.ctypes.data_as(ctypes.c_void_p),
            M, K, N
        )
        _rt.lib.cpu_relu(
            out_arr.ctypes.data_as(ctypes.c_void_p),
            out_arr.ctypes.data_as(ctypes.c_void_p),
            M * N
        )
        return Tensor(out_arr, device=Device.CPU)

def matmul_add(A, B, bias):
    if hasattr(A, "graph") or hasattr(B, "graph") or hasattr(bias, "graph"):
        return A @ B + bias
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)
    t_bias = _to_tensor(bias)
    M, K = t_A.shape
    _, N = t_B.shape
    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if use_gpu:
        g_A = t_A.to(Device.GPU)
        g_B = t_B.to(Device.GPU)
        g_bias = t_bias.to(Device.GPU)
        C_ptr = _rt.mempool.allocate(M * N * 4, _rt.lib)
        _rt.lib.forge_gpu_matmul_add_fused(g_A.gpu_ptr, g_B.gpu_ptr, g_bias.gpu_ptr, C_ptr, M, K, N)
        _rt.lib.forge_gpu_sync()
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, M * N * 4)
        _rt.mempool.free(C_ptr, M * N * 4)
        return Tensor(out_arr, device=Device.GPU)
    else:
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.cpu_matmul_add(
            t_A.numpy().ctypes.data_as(ctypes.c_void_p),
            t_B.numpy().ctypes.data_as(ctypes.c_void_p),
            t_bias.numpy().ctypes.data_as(ctypes.c_void_p),
            out_arr.ctypes.data_as(ctypes.c_void_p),
            M, K, N
        )
        return Tensor(out_arr, device=Device.CPU)

def matmul_add_relu(A, B, bias):
    if hasattr(A, "graph") or hasattr(B, "graph") or hasattr(bias, "graph"):
        return (A @ B + bias).relu()
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)
    t_bias = _to_tensor(bias)
    M, K = t_A.shape
    _, N = t_B.shape
    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if use_gpu:
        g_A = t_A.to(Device.GPU)
        g_B = t_B.to(Device.GPU)
        g_bias = t_bias.to(Device.GPU)
        C_ptr = _rt.mempool.allocate(M * N * 4, _rt.lib)
        _rt.lib.forge_gpu_matmul_add_relu_fused(g_A.gpu_ptr, g_B.gpu_ptr, g_bias.gpu_ptr, C_ptr, M, K, N)
        _rt.lib.forge_gpu_sync()
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, M * N * 4)
        _rt.mempool.free(C_ptr, M * N * 4)
        return Tensor(out_arr, device=Device.GPU)
    else:
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.cpu_matmul_add_relu(
            t_A.numpy().ctypes.data_as(ctypes.c_void_p),
            t_B.numpy().ctypes.data_as(ctypes.c_void_p),
            t_bias.numpy().ctypes.data_as(ctypes.c_void_p),
            out_arr.ctypes.data_as(ctypes.c_void_p),
            M, K, N
        )
        return Tensor(out_arr, device=Device.CPU)

def matmul_add_relu_residual(A, B, bias, residual):
    if hasattr(A, "graph") or hasattr(B, "graph") or hasattr(bias, "graph") or hasattr(residual, "graph"):
        return (A @ B + bias).relu() + residual
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)
    t_bias = _to_tensor(bias)
    t_res = _to_tensor(residual)
    M, K = t_A.shape
    _, N = t_B.shape
    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if use_gpu:
        g_A = t_A.to(Device.GPU)
        g_B = t_B.to(Device.GPU)
        g_bias = t_bias.to(Device.GPU)
        g_res = t_res.to(Device.GPU)
        C_ptr = _rt.mempool.allocate(M * N * 4, _rt.lib)
        _rt.lib.forge_gpu_matmul_add_relu_residual_fused(g_A.gpu_ptr, g_B.gpu_ptr, g_bias.gpu_ptr, g_res.gpu_ptr, C_ptr, M, K, N)
        _rt.lib.forge_gpu_sync()
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, M * N * 4)
        _rt.mempool.free(C_ptr, M * N * 4)
        return Tensor(out_arr, device=Device.GPU)
    else:
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.cpu_matmul_add_relu_residual(
            t_A.numpy().ctypes.data_as(ctypes.c_void_p),
            t_B.numpy().ctypes.data_as(ctypes.c_void_p),
            t_bias.numpy().ctypes.data_as(ctypes.c_void_p),
            t_res.numpy().ctypes.data_as(ctypes.c_void_p),
            out_arr.ctypes.data_as(ctypes.c_void_p),
            M, K, N
        )
        return Tensor(out_arr, device=Device.CPU)

def add_relu(A, B):
    if hasattr(A, "graph") or hasattr(B, "graph"):
        return (A + B).relu()
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)
    n = int(np.prod(t_A.shape))
    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if use_gpu:
        g_A = t_A.to(Device.GPU)
        g_B = t_B.to(Device.GPU)
        C_ptr = _rt.mempool.allocate(n * 4, _rt.lib)
        _rt.lib.forge_gpu_add_relu_fused(g_A.gpu_ptr, g_B.gpu_ptr, C_ptr, n)
        _rt.lib.forge_gpu_sync()
        out_arr = np.empty(t_A.shape, dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, n * 4)
        _rt.mempool.free(C_ptr, n * 4)
        return Tensor(out_arr, device=Device.GPU)
    else:
        out_arr = np.empty(t_A.shape, dtype=np.float32)
        _rt.lib.cpu_add_relu(
            t_A.numpy().ctypes.data_as(ctypes.c_void_p),
            t_B.numpy().ctypes.data_as(ctypes.c_void_p),
            out_arr.ctypes.data_as(ctypes.c_void_p),
            n
        )
        return Tensor(out_arr, device=Device.CPU)

def memory_stats() -> Dict[str, int]:
    return _rt.mempool.stats()

def memory_clear():
    if _rt.lib:
        _rt.mempool.clear(_rt.lib)


