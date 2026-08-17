"""
Qualcomm Hardware Test Suite: Hexagon NPU (HTP) and Adreno GPU (QNN GPU).
"""

import os
import sys
import ctypes
import numpy as np

# Use the ARM64 compiled pjrt_qnn.dll
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "pjrt_qnn.dll"))
if not os.path.exists(dll_path):
    dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jax_plugins", "qnn", "pjrt_qnn.dll"))

print(f"Loading native library: {dll_path}")
lib = ctypes.CDLL(dll_path)

# Function Signatures
lib.JAX_QNN_Init.restype = ctypes.c_int
lib.JAX_QNN_Init.argtypes = [ctypes.c_int, ctypes.c_char_p]

lib.JAX_QNN_GetBackendName.restype = ctypes.c_char_p
lib.JAX_QNN_GetBackendName.argtypes = []

lib.JAX_QNN_Compile.restype = ctypes.c_void_p
lib.JAX_QNN_Compile.argtypes = [ctypes.c_char_p, ctypes.c_size_t]

lib.JAX_QNN_DestroyExecutable.restype = None
lib.JAX_QNN_DestroyExecutable.argtypes = [ctypes.c_void_p]

lib.JAX_QNN_GetNumInputs.restype = ctypes.c_int
lib.JAX_QNN_GetNumInputs.argtypes = [ctypes.c_void_p]

lib.JAX_QNN_GetNumOutputs.restype = ctypes.c_int
lib.JAX_QNN_GetNumOutputs.argtypes = [ctypes.c_void_p]

lib.JAX_QNN_Execute.restype = ctypes.c_int
lib.JAX_QNN_Execute.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int,
]

sdk_root = os.environ.get(
    "QNN_SDK_ROOT",
    r"C:\Users\rawat\Downloads\v2.37.0.250724\qairt\2.37.0.250724"
).encode("utf-8")

def test_target(target_name: str, target_id: int):
    print(f"\n=======================================================")
    print(f" Testing Qualcomm Target: {target_name} (ID={target_id})")
    print(f"=======================================================")
    
    # 1. Initialize Hardware Backend
    res = lib.JAX_QNN_Init(target_id, sdk_root)
    backend_str = lib.JAX_QNN_GetBackendName().decode("utf-8")
    print(f"[+] Init Result: {res} (Backend: {backend_str})")

    # 2. Compile a GEMM + ReLU computation
    mlir_program = """
func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  } : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = "stablehlo.maximum"(%0, %0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
""".strip().encode("utf-8")

    exec_handle = lib.JAX_QNN_Compile(mlir_program, len(mlir_program))
    if not exec_handle:
        print(f"[-] Compilation failed for target {target_name}")
        return False

    print(f"[+] Graph Compiled Successfully (Handle: {hex(exec_handle)})")
    print(f"[+] Inputs: {lib.JAX_QNN_GetNumInputs(exec_handle)}, Outputs: {lib.JAX_QNN_GetNumOutputs(exec_handle)}")

    # 3. Create input tensors
    a = np.ones((4, 4), dtype=np.float32) * 2.0
    b = np.ones((4, 4), dtype=np.float32) * 3.0
    out = np.zeros((4, 4), dtype=np.float32)

    in_ptrs = (ctypes.c_void_p * 2)(a.ctypes.data, b.ctypes.data)
    in_bytes = (ctypes.c_int64 * 2)(a.nbytes, b.nbytes)
    out_ptrs = (ctypes.c_void_p * 1)(out.ctypes.data)
    out_bytes = (ctypes.c_int64 * 1)(out.nbytes)

    # 4. Execute on Qualcomm Hardware
    status = lib.JAX_QNN_Execute(exec_handle, in_ptrs, in_bytes, 2, out_ptrs, out_bytes, 1)
    print(f"[+] Hardware Execution Status: {status}")

    # Expected: 4 * (2.0 * 3.0) = 24.0
    expected = np.matmul(a, b)
    assert np.allclose(out, expected), f"Mismatch! Got {out}, expected {expected}"
    print(f"[+] Correctness Verified! Output[0, 0] = {out[0, 0]} (Expected: {expected[0, 0]})")

    lib.JAX_QNN_DestroyExecutable(exec_handle)
    return True

if __name__ == "__main__":
    # Test 1: Qualcomm Hexagon NPU (HTP - ID=1)
    htp_ok = test_target("Qualcomm Hexagon NPU (HTP)", 1)

    # Test 2: Qualcomm Adreno GPU (GPU - ID=2)
    gpu_ok = test_target("Qualcomm Adreno GPU (OpenCL/Direct3D)", 2)
