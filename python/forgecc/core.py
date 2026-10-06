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
        self.lib.forge_has_gpu_support.restype = ctypes.c_bool
        self.lib.forge_has_gpu_support.argtypes = []

        self.lib.forge_sync.restype = None
        self.lib.forge_sync.argtypes = []

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

        self.lib.forge_gpu_matmul_relu_fused.restype = None
        self.lib.forge_gpu_matmul_relu_fused.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
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

_rt = ForgeRuntime()

def is_gpu_available() -> bool:
    if _rt.lib:
        try:
            return bool(_rt.lib.forge_has_gpu_support())
        except Exception:
            return False
    return False

def sync():
    if _rt.lib:
        _rt.lib.forge_sync()

class Tensor:
    def __init__(self, data, device=Device.CPU):
        if isinstance(data, np.ndarray):
            self.host_array = np.ascontiguousarray(data, dtype=np.float32)
        elif isinstance(data, (list, tuple)):
            self.host_array = np.ascontiguousarray(data, dtype=np.float32)
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
            new_t = Tensor(self.numpy(), device=Device.GPU)
            return new_t
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

def tensor(data, device=Device.CPU) -> Tensor:
    return Tensor(data, device=device)

def matmul(A, B, dev=None) -> np.ndarray:
    if isinstance(A, Tensor) and isinstance(B, Tensor):
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Shape mismatch: {A.shape} and {B.shape}")
        M, K = A.shape
        _, N = B.shape
        if A.device == Device.GPU and B.device == Device.GPU and is_gpu_available():
            C_gpu = _rt.lib.forge_gpu_malloc(M * N * 4)
            _rt.lib.forge_gpu_matmul(A.gpu_ptr, B.gpu_ptr, C_gpu, M, K, N)
            _rt.lib.forge_sync()
            out = np.empty((M, N), dtype=np.float32)
            _rt.lib.forge_gpu_memcpy_to_host(out.ctypes.data_as(ctypes.c_void_p), C_gpu, M * N * 4)
            _rt.lib.forge_gpu_free(C_gpu)
            return out
        return np.matmul(A.numpy(), B.numpy())

    A_f32 = np.ascontiguousarray(A, dtype=np.float32)
    B_f32 = np.ascontiguousarray(B, dtype=np.float32)
    if A_f32.ndim != 2 or B_f32.ndim != 2:
        raise ValueError("Inputs must be 2D matrices.")
    if A_f32.shape[1] != B_f32.shape[0]:
        raise ValueError(f"Shape mismatch: {A_f32.shape} and {B_f32.shape}")

    M, K = A_f32.shape
    _, N = B_f32.shape

    target_dev = Device.GPU if (dev == Device.GPU or (dev is None and is_gpu_available())) else Device.CPU
    if target_dev == Device.GPU and is_gpu_available():
        bytes_A = A_f32.nbytes
        bytes_B = B_f32.nbytes
        bytes_C = M * N * 4

        d_A = _rt.lib.forge_gpu_malloc(bytes_A)
        d_B = _rt.lib.forge_gpu_malloc(bytes_B)
        d_C = _rt.lib.forge_gpu_malloc(bytes_C)

        _rt.lib.forge_gpu_memcpy_to_device(d_A, A_f32.ctypes.data_as(ctypes.c_void_p), bytes_A)
        _rt.lib.forge_gpu_memcpy_to_device(d_B, B_f32.ctypes.data_as(ctypes.c_void_p), bytes_B)

        _rt.lib.forge_gpu_matmul(d_A, d_B, d_C, ctypes.c_int64(M), ctypes.c_int64(K), ctypes.c_int64(N))
        _rt.lib.forge_sync()

        C = np.empty((M, N), dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(C.ctypes.data_as(ctypes.c_void_p), d_C, bytes_C)

        _rt.lib.forge_gpu_free(d_A)
        _rt.lib.forge_gpu_free(d_B)
        _rt.lib.forge_gpu_free(d_C)
        return C
    else:
        return np.matmul(A_f32, B_f32)

def relu(x, dev=None) -> np.ndarray:
    if isinstance(x, Tensor):
        if x.device == Device.GPU and is_gpu_available():
            n = int(np.prod(x.shape))
            C_gpu = _rt.lib.forge_gpu_malloc(n * 4)
            _rt.lib.forge_gpu_relu(x.gpu_ptr, C_gpu, n)
            _rt.lib.forge_sync()
            out = np.empty(x.shape, dtype=np.float32)
            _rt.lib.forge_gpu_memcpy_to_host(out.ctypes.data_as(ctypes.c_void_p), C_gpu, n * 4)
            _rt.lib.forge_gpu_free(C_gpu)
            return out
        return np.maximum(0, x.numpy())

    x_f32 = np.ascontiguousarray(x, dtype=np.float32)
    n = x_f32.size

    target_dev = Device.GPU if (dev == Device.GPU or (dev is None and is_gpu_available())) else Device.CPU
    if target_dev == Device.GPU and is_gpu_available():
        bytes_x = x_f32.nbytes
        d_in = _rt.lib.forge_gpu_malloc(bytes_x)
        d_out = _rt.lib.forge_gpu_malloc(bytes_x)

        _rt.lib.forge_gpu_memcpy_to_device(d_in, x_f32.ctypes.data_as(ctypes.c_void_p), bytes_x)
        _rt.lib.forge_gpu_relu(d_in, d_out, ctypes.c_int64(n))
        _rt.lib.forge_sync()

        out = np.empty_like(x_f32)
        _rt.lib.forge_gpu_memcpy_to_host(out.ctypes.data_as(ctypes.c_void_p), d_out, bytes_x)

        _rt.lib.forge_gpu_free(d_in)
        _rt.lib.forge_gpu_free(d_out)
        return out
    else:
        return np.maximum(0, x_f32)

def matmul_relu(A, B, dev=None) -> np.ndarray:
    if isinstance(A, Tensor) and isinstance(B, Tensor):
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Shape mismatch: {A.shape} and {B.shape}")
        M, K = A.shape
        _, N = B.shape
        if A.device == Device.GPU and B.device == Device.GPU and is_gpu_available():
            C_gpu = _rt.lib.forge_gpu_malloc(M * N * 4)
            _rt.lib.forge_gpu_matmul_relu_fused(A.gpu_ptr, B.gpu_ptr, C_gpu, M, K, N)
            _rt.lib.forge_sync()
            out = np.empty((M, N), dtype=np.float32)
            _rt.lib.forge_gpu_memcpy_to_host(out.ctypes.data_as(ctypes.c_void_p), C_gpu, M * N * 4)
            _rt.lib.forge_gpu_free(C_gpu)
            return out
        return np.maximum(0, np.matmul(A.numpy(), B.numpy()))

    A_f32 = np.ascontiguousarray(A, dtype=np.float32)
    B_f32 = np.ascontiguousarray(B, dtype=np.float32)
    if A_f32.ndim != 2 or B_f32.ndim != 2:
        raise ValueError("Inputs must be 2D matrices.")
    if A_f32.shape[1] != B_f32.shape[0]:
        raise ValueError(f"Shape mismatch: {A_f32.shape} and {B_f32.shape}")

    M, K = A_f32.shape
    _, N = B_f32.shape

    target_dev = Device.GPU if (dev == Device.GPU or (dev is None and is_gpu_available())) else Device.CPU
    if target_dev == Device.GPU and is_gpu_available():
        bytes_A = A_f32.nbytes
        bytes_B = B_f32.nbytes
        bytes_C = M * N * 4

        d_A = _rt.lib.forge_gpu_malloc(bytes_A)
        d_B = _rt.lib.forge_gpu_malloc(bytes_B)
        d_C = _rt.lib.forge_gpu_malloc(bytes_C)

        _rt.lib.forge_gpu_memcpy_to_device(d_A, A_f32.ctypes.data_as(ctypes.c_void_p), bytes_A)
        _rt.lib.forge_gpu_memcpy_to_device(d_B, B_f32.ctypes.data_as(ctypes.c_void_p), bytes_B)

        _rt.lib.forge_gpu_matmul_relu_fused(d_A, d_B, d_C, ctypes.c_int64(M), ctypes.c_int64(K), ctypes.c_int64(N))
        _rt.lib.forge_sync()

        C = np.empty((M, N), dtype=np.float32)
        _rt.lib.forge_gpu_memcpy_to_host(C.ctypes.data_as(ctypes.c_void_p), d_C, bytes_C)

        _rt.lib.forge_gpu_free(d_A)
        _rt.lib.forge_gpu_free(d_B)
        _rt.lib.forge_gpu_free(d_C)
        return C
    else:
        return np.maximum(0, np.matmul(A_f32, B_f32))
