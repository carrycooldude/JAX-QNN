# =============================================================================
# jax_qnn/bridge.py — Direct Native Ctypes Bridge to pjrt_qnn.dll
# =============================================================================

"""
Direct ctypes interface to the compiled Qualcomm QNN execution engine (pjrt_qnn.dll).
"""

import os
import sys
import ctypes
import numpy as np
from typing import List, Tuple, Optional

import platform

def _load_qnn_dll():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(dir_path, "..", "build", "pjrt_qnn.dll"),
        os.path.join(dir_path, "..", "build", "Release", "pjrt_qnn.dll"),
        os.path.join(dir_path, "..", "build_x64", "Release", "pjrt_qnn.dll"),
        os.path.join(dir_path, "..", "build_x64", "pjrt_qnn.dll"),
        os.path.join(dir_path, "pjrt_qnn.dll"),
        os.path.join(dir_path, "..", "jax_plugins", "qnn", "pjrt_qnn.dll"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                lib = ctypes.CDLL(os.path.abspath(path))
                return lib, os.path.abspath(path)
            except Exception:
                continue
    return None, ""

_LIB, _DLL_PATH = _load_qnn_dll()

if _LIB is not None:
    try:
        # JAX_QNN_Init(int backend_type, const char* sdk_root)
        _LIB.JAX_QNN_Init.restype = ctypes.c_int
        _LIB.JAX_QNN_Init.argtypes = [ctypes.c_int, ctypes.c_char_p]

        # JAX_QNN_GetBackendName()
        _LIB.JAX_QNN_GetBackendName.restype = ctypes.c_char_p
        _LIB.JAX_QNN_GetBackendName.argtypes = []

        # JAX_QNN_Compile(const char* mlir_text, size_t text_len)
        _LIB.JAX_QNN_Compile.restype = ctypes.c_void_p
        _LIB.JAX_QNN_Compile.argtypes = [ctypes.c_char_p, ctypes.c_size_t]

        # JAX_QNN_DestroyExecutable(void* exec_handle)
        _LIB.JAX_QNN_DestroyExecutable.restype = None
        _LIB.JAX_QNN_DestroyExecutable.argtypes = [ctypes.c_void_p]

        # JAX_QNN_GetNumInputs(void* exec_handle)
        _LIB.JAX_QNN_GetNumInputs.restype = ctypes.c_int
        _LIB.JAX_QNN_GetNumInputs.argtypes = [ctypes.c_void_p]

        # JAX_QNN_GetNumOutputs(void* exec_handle)
        _LIB.JAX_QNN_GetNumOutputs.restype = ctypes.c_int
        _LIB.JAX_QNN_GetNumOutputs.argtypes = [ctypes.c_void_p]

        # JAX_QNN_GetOutputByteSize(void* exec_handle, int output_idx)
        _LIB.JAX_QNN_GetOutputByteSize.restype = ctypes.c_int64
        _LIB.JAX_QNN_GetOutputByteSize.argtypes = [ctypes.c_void_p, ctypes.c_int]

        # JAX_QNN_GetOutputRank(void* exec_handle, int output_idx)
        _LIB.JAX_QNN_GetOutputRank.restype = ctypes.c_int
        _LIB.JAX_QNN_GetOutputRank.argtypes = [ctypes.c_void_p, ctypes.c_int]

        # JAX_QNN_GetOutputShape(void* exec_handle, int output_idx, int64_t* dims_out)
        _LIB.JAX_QNN_GetOutputShape.restype = ctypes.c_int
        _LIB.JAX_QNN_GetOutputShape.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
        ]

        # JAX_QNN_Execute(...)
        _LIB.JAX_QNN_Execute.restype = ctypes.c_int
        _LIB.JAX_QNN_Execute.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int,
        ]

        # Initialize QNN SDK backend
        sdk_root = os.environ.get("QNN_SDK_ROOT", "").encode("utf-8")
        _LIB.JAX_QNN_Init(0, sdk_root if sdk_root else None)
    except Exception as e:
        _LIB = None


def is_native_available() -> bool:
    """Returns True if pjrt_qnn.dll is loaded and initialized."""
    return _LIB is not None


def get_backend_name() -> str:
    """Returns the name of the active QNN hardware/CPU backend."""
    if _LIB is None:
        return "none"
    res = _LIB.JAX_QNN_GetBackendName()
    return res.decode("utf-8") if res else "unknown"


class NativeQnnExecutable:
    """Wrapped handle to a compiled native QNN computation."""

    def __init__(self, handle: int):
        self._handle = handle
        self.num_inputs = _LIB.JAX_QNN_GetNumInputs(handle)
        self.num_outputs = _LIB.JAX_QNN_GetNumOutputs(handle)
        
        # Pre-compute output shapes
        self.output_shapes = []
        for i in range(self.num_outputs):
            rank = _LIB.JAX_QNN_GetOutputRank(handle, i)
            dims = (ctypes.c_int64 * rank)()
            _LIB.JAX_QNN_GetOutputShape(handle, i, dims)
            self.output_shapes.append(tuple(dims[j] for j in range(rank)))

    def __call__(self, *args) -> List[np.ndarray]:
        """Executes computation with NumPy/JAX array arguments."""
        if not self._handle or _LIB is None:
            raise RuntimeError("Executable is not compiled or library missing.")

        # Convert args to contiguous arrays
        np_args = [np.ascontiguousarray(a) for a in args]
        num_inputs = len(np_args)

        in_ptrs = (ctypes.c_void_p * num_inputs)()
        in_bytes = (ctypes.c_int64 * num_inputs)()

        for i, a in enumerate(np_args):
            in_ptrs[i] = a.ctypes.data
            in_bytes[i] = a.nbytes

        # Allocate output arrays
        out_arrays = []
        out_ptrs = (ctypes.c_void_p * self.num_outputs)()
        out_bytes = (ctypes.c_int64 * self.num_outputs)()

        for i in range(self.num_outputs):
            shape = self.output_shapes[i] if i < len(self.output_shapes) else ()
            arr = np.empty(shape, dtype=np.float32)
            out_arrays.append(arr)
            out_ptrs[i] = arr.ctypes.data
            out_bytes[i] = arr.nbytes

        ret = _LIB.JAX_QNN_Execute(
            self._handle,
            in_ptrs,
            in_bytes,
            num_inputs,
            out_ptrs,
            out_bytes,
            self.num_outputs,
        )

        if ret != 0:
            raise RuntimeError(f"QNN execution failed with error code: {ret}")

        return out_arrays

    def __del__(self):
        if self._handle and _LIB is not None:
            try:
                _LIB.JAX_QNN_DestroyExecutable(self._handle)
            except Exception:
                pass
            self._handle = None


def compile_stablehlo(mlir_text: str) -> NativeQnnExecutable:
    """Compiles MLIR StableHLO string into a native QnnExecutable."""
    if _LIB is None:
        raise RuntimeError("pjrt_qnn native library is not available.")

    encoded = mlir_text.encode("utf-8")
    handle = _LIB.JAX_QNN_Compile(encoded, len(encoded))
    if not handle:
        raise ValueError(f"Failed to compile StableHLO code to QNN graph.")

    return NativeQnnExecutable(handle)
