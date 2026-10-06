import os
import sys
import ctypes
import numpy as np
from pathlib import Path

def _find_library():
    # Look for compiled dll/so in build or local package directory
    current_dir = Path(__file__).parent.resolve()
    candidates = [
        current_dir / "libforgecc_rt.dll",
        current_dir / "libforgecc_rt.so",
        current_dir.parent.parent / "build" / "runtime" / "libforgecc_rt.a",
        current_dir.parent.parent / "build" / "runtime" / "libforgecc_rt.dll",
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
        lib_path = _find_library()
        if not lib_path:
            self.lib = None
            return
        
        try:
            self.lib = ctypes.CDLL(lib_path)
            self._setup_signatures()
        except Exception:
            self.lib = None

    def _setup_signatures(self):
        self.lib.forge_has_gpu_support.restype = ctypes.c_bool
        self.lib.forge_has_gpu_support.argtypes = []

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

        self.lib.forge_sync.restype = None
        self.lib.forge_sync.argtypes = []

_rt = ForgeRuntime()

def is_gpu_available() -> bool:
    if _rt.lib:
        return bool(_rt.lib.forge_has_gpu_support())
    return False

def matmul_relu(A: np.ndarray, B: np.ndarray, dev: int = Device.GPU) -> np.ndarray:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Inputs must be 2D matrices.")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Shape mismatch: {A.shape} and {B.shape}")

    A_f32 = np.ascontiguousarray(A, dtype=np.float32)
    B_f32 = np.ascontiguousarray(B, dtype=np.float32)
    
    M, K = A_f32.shape
    K2, N = B_f32.shape
    C = np.zeros((M, N), dtype=np.float32)

    if _rt.lib and dev == Device.GPU and is_gpu_available():
        _rt.lib.forge_gpu_matmul_relu_fused(
            A_f32.ctypes.data_as(ctypes.c_void_p),
            B_f32.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(M), ctypes.c_int64(K), ctypes.c_int64(N)
        )
        _rt.lib.forge_sync()
    else:
        # Fallback numpy compute
        C = np.maximum(0, np.matmul(A_f32, B_f32))

    return C
