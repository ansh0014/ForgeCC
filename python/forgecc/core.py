import os
import sys
import ctypes
import numpy as np
from pathlib import Path

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

class ForgeRuntime:
    def __init__(self):
        self.lib = None
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

class Tensor:
    def __init__(self, data, device=Device.CPU):
        if isinstance(data, np.ndarray):
            self.host_array = np.ascontiguousarray(data, dtype=np.float32)
        elif isinstance(data, (list, tuple, float, int)):
            self.host_array = np.ascontiguousarray(np.array(data, dtype=np.float32))
        elif isinstance(data, Tensor):
            self.host_array = data.numpy().copy()
        else:
            raise TypeError("Unsupported data type for Tensor initialization.")

        self.shape = self.host_array.shape
        self.device = device
        self.gpu_ptr = None
        self.nbytes = self.host_array.nbytes

        if self.device == Device.GPU:
            if not is_gpu_available():
                self.device = Device.CPU
            else:
                self.gpu_ptr = _rt.lib.forge_gpu_malloc(self.nbytes)
                if self.gpu_ptr:
                    _rt.lib.forge_gpu_memcpy_to_device(
                        self.gpu_ptr,
                        self.host_array.ctypes.data_as(ctypes.c_void_p),
                        self.nbytes
                    )
                else:
                    self.device = Device.CPU

    def __del__(self):
        if self.gpu_ptr and _rt.lib:
            try:
                _rt.lib.forge_gpu_free(self.gpu_ptr)
            except Exception:
                pass
            self.gpu_ptr = None

    def to(self, target_device):
        if self.device == target_device:
            return self

        if target_device == Device.GPU and is_gpu_available():
            return Tensor(self.numpy(), device=Device.GPU)
        else:
            return Tensor(self.numpy(), device=Device.CPU)

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

def tensor(data, device=Device.CPU) -> Tensor:
    return Tensor(data, device=device)

def _to_tensor(x, device=Device.CPU):
    if isinstance(x, Tensor):
        return x
    return Tensor(x, device=device)

def matmul(A, B) -> Tensor:
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
        C_ptr = _rt.lib.forge_gpu_malloc(M * N * 4)
        _rt.lib.forge_gpu_matmul(g_A.gpu_ptr, g_B.gpu_ptr, C_ptr, M, K, N)
        _rt.lib.forge_gpu_sync()
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, M * N * 4)
        _rt.lib.forge_gpu_free(C_ptr)
        return Tensor(out_arr, device=Device.GPU)
    else:
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.cpu_matmul(
            t_A.numpy().ctypes.data_as(ctypes.c_void_p),
            t_B.numpy().ctypes.data_as(ctypes.c_void_p),
            out_arr.ctypes.data_as(ctypes.c_void_p),
            M, K, N
        )
        return Tensor(out_arr, device=Device.CPU)

def add(A, B) -> Tensor:
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)

    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if t_A.shape == t_B.shape:
        n = int(np.prod(t_A.shape))
        if use_gpu:
            g_A = t_A.to(Device.GPU)
            g_B = t_B.to(Device.GPU)
            C_ptr = _rt.lib.forge_gpu_malloc(n * 4)
            _rt.lib.forge_gpu_add(g_A.gpu_ptr, g_B.gpu_ptr, C_ptr, n)
            _rt.lib.forge_gpu_sync()
            out_arr = np.empty(t_A.shape, dtype=np.float32)
            _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, n * 4)
            _rt.lib.forge_gpu_free(C_ptr)
            return Tensor(out_arr, device=Device.GPU)
        else:
            out_arr = np.empty(t_A.shape, dtype=np.float32)
            _rt.lib.cpu_add(
                t_A.numpy().ctypes.data_as(ctypes.c_void_p),
                t_B.numpy().ctypes.data_as(ctypes.c_void_p),
                out_arr.ctypes.data_as(ctypes.c_void_p),
                n
            )
            return Tensor(out_arr, device=Device.CPU)
    else:
        res = t_A.numpy() + t_B.numpy()
        return Tensor(res, device=Device.GPU if use_gpu else Device.CPU)

def sub(A, B) -> Tensor:
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)
    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if t_A.shape == t_B.shape:
        n = int(np.prod(t_A.shape))
        if use_gpu:
            g_A = t_A.to(Device.GPU)
            g_B = t_B.to(Device.GPU)
            C_ptr = _rt.lib.forge_gpu_malloc(n * 4)
            _rt.lib.forge_gpu_sub(g_A.gpu_ptr, g_B.gpu_ptr, C_ptr, n)
            _rt.lib.forge_gpu_sync()
            out_arr = np.empty(t_A.shape, dtype=np.float32)
            _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, n * 4)
            _rt.lib.forge_gpu_free(C_ptr)
            return Tensor(out_arr, device=Device.GPU)
        else:
            out_arr = np.empty(t_A.shape, dtype=np.float32)
            _rt.lib.cpu_sub(
                t_A.numpy().ctypes.data_as(ctypes.c_void_p),
                t_B.numpy().ctypes.data_as(ctypes.c_void_p),
                out_arr.ctypes.data_as(ctypes.c_void_p),
                n
            )
            return Tensor(out_arr, device=Device.CPU)
    else:
        res = t_A.numpy() - t_B.numpy()
        return Tensor(res, device=Device.GPU if use_gpu else Device.CPU)

def mul(A, B) -> Tensor:
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)
    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if t_A.shape == t_B.shape:
        n = int(np.prod(t_A.shape))
        if use_gpu:
            g_A = t_A.to(Device.GPU)
            g_B = t_B.to(Device.GPU)
            C_ptr = _rt.lib.forge_gpu_malloc(n * 4)
            _rt.lib.forge_gpu_mul(g_A.gpu_ptr, g_B.gpu_ptr, C_ptr, n)
            _rt.lib.forge_gpu_sync()
            out_arr = np.empty(t_A.shape, dtype=np.float32)
            _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, n * 4)
            _rt.lib.forge_gpu_free(C_ptr)
            return Tensor(out_arr, device=Device.GPU)
        else:
            out_arr = np.empty(t_A.shape, dtype=np.float32)
            _rt.lib.cpu_mul(
                t_A.numpy().ctypes.data_as(ctypes.c_void_p),
                t_B.numpy().ctypes.data_as(ctypes.c_void_p),
                out_arr.ctypes.data_as(ctypes.c_void_p),
                n
            )
            return Tensor(out_arr, device=Device.CPU)
    else:
        res = t_A.numpy() * t_B.numpy()
        return Tensor(res, device=Device.GPU if use_gpu else Device.CPU)

def div(A, B) -> Tensor:
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)
    res = np.divide(t_A.numpy(), t_B.numpy(), out=np.zeros_like(t_A.numpy()), where=(t_B.numpy() != 0))
    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    return Tensor(res, device=Device.GPU if use_gpu else Device.CPU)

def relu(x) -> Tensor:
    t_x = _to_tensor(x)
    n = int(np.prod(t_x.shape))
    if t_x.device == Device.GPU and is_gpu_available():
        C_ptr = _rt.lib.forge_gpu_malloc(n * 4)
        _rt.lib.forge_gpu_relu(t_x.gpu_ptr, C_ptr, n)
        _rt.lib.forge_gpu_sync()
        out_arr = np.empty(t_x.shape, dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, n * 4)
        _rt.lib.forge_gpu_free(C_ptr)
        return Tensor(out_arr, device=Device.GPU)
    else:
        out_arr = np.empty(t_x.shape, dtype=np.float32)
        _rt.lib.cpu_relu(
            t_x.numpy().ctypes.data_as(ctypes.c_void_p),
            out_arr.ctypes.data_as(ctypes.c_void_p),
            n
        )
        return Tensor(out_arr, device=Device.CPU)

def sigmoid(x) -> Tensor:
    t_x = _to_tensor(x)
    n = int(np.prod(t_x.shape))
    out_arr = np.empty(t_x.shape, dtype=np.float32)
    _rt.lib.cpu_sigmoid(
        t_x.numpy().ctypes.data_as(ctypes.c_void_p),
        out_arr.ctypes.data_as(ctypes.c_void_p),
        n
    )
    return Tensor(out_arr, device=t_x.device)

def gelu(x) -> Tensor:
    t_x = _to_tensor(x)
    n = int(np.prod(t_x.shape))
    out_arr = np.empty(t_x.shape, dtype=np.float32)
    _rt.lib.cpu_gelu(
        t_x.numpy().ctypes.data_as(ctypes.c_void_p),
        out_arr.ctypes.data_as(ctypes.c_void_p),
        n
    )
    return Tensor(out_arr, device=t_x.device)

def tanh(x) -> Tensor:
    t_x = _to_tensor(x)
    n = int(np.prod(t_x.shape))
    out_arr = np.empty(t_x.shape, dtype=np.float32)
    _rt.lib.cpu_tanh_act(
        t_x.numpy().ctypes.data_as(ctypes.c_void_p),
        out_arr.ctypes.data_as(ctypes.c_void_p),
        n
    )
    return Tensor(out_arr, device=t_x.device)

def sum_tensor(x) -> Tensor:
    t_x = _to_tensor(x)
    n = int(np.prod(t_x.shape))
    res = np.empty((1,), dtype=np.float32)
    _rt.lib.cpu_sum(
        t_x.numpy().ctypes.data_as(ctypes.c_void_p),
        res.ctypes.data_as(ctypes.c_void_p),
        n
    )
    return Tensor(res, device=t_x.device)

def mean_tensor(x) -> Tensor:
    t_x = _to_tensor(x)
    n = int(np.prod(t_x.shape))
    res = np.empty((1,), dtype=np.float32)
    _rt.lib.cpu_mean(
        t_x.numpy().ctypes.data_as(ctypes.c_void_p),
        res.ctypes.data_as(ctypes.c_void_p),
        n
    )
    return Tensor(res, device=t_x.device)

def matmul_relu(A, B) -> Tensor:
    t_A = _to_tensor(A)
    t_B = _to_tensor(B)
    M, K = t_A.shape
    _, N = t_B.shape
    use_gpu = (t_A.device == Device.GPU or t_B.device == Device.GPU) and is_gpu_available()
    if use_gpu:
        g_A = t_A.to(Device.GPU)
        g_B = t_B.to(Device.GPU)
        C_ptr = _rt.lib.forge_gpu_malloc(M * N * 4)
        _rt.lib.forge_gpu_matmul_relu_fused(g_A.gpu_ptr, g_B.gpu_ptr, C_ptr, M, K, N)
        _rt.lib.forge_gpu_sync()
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, M * N * 4)
        _rt.lib.forge_gpu_free(C_ptr)
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

def matmul_add(A, B, bias) -> Tensor:
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
        C_ptr = _rt.lib.forge_gpu_malloc(M * N * 4)
        _rt.lib.forge_gpu_matmul_add_fused(g_A.gpu_ptr, g_B.gpu_ptr, g_bias.gpu_ptr, C_ptr, M, K, N)
        _rt.lib.forge_gpu_sync()
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, M * N * 4)
        _rt.lib.forge_gpu_free(C_ptr)
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

def matmul_add_relu(A, B, bias) -> Tensor:
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
        C_ptr = _rt.lib.forge_gpu_malloc(M * N * 4)
        _rt.lib.forge_gpu_matmul_add_relu_fused(g_A.gpu_ptr, g_B.gpu_ptr, g_bias.gpu_ptr, C_ptr, M, K, N)
        _rt.lib.forge_gpu_sync()
        out_arr = np.empty((M, N), dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(out_arr.ctypes.data_as(ctypes.c_void_p), C_ptr, M * N * 4)
        _rt.lib.forge_gpu_free(C_ptr)
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

