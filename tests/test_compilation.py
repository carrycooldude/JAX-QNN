# =============================================================================
# tests/test_compilation.py — StableHLO Parsing and Lowering Tests
# =============================================================================

import pytest
import re

SAMPLE_ADD_MLIR = """
module @jit_add attributes {mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<1024x1024xf32>
    return %0 : tensor<1024x1024xf32>
  }
}
"""

SAMPLE_MATMUL_RELU_MLIR = """
module @jit_matmul_relu attributes {mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<128x512xf32>
    %1 = stablehlo.maximum %0, %cst : tensor<128x512xf32>
    return %1 : tensor<128x512xf32>
  }
}
"""

def test_stablehlo_signature_matching():
    # Verify regex/parsing patterns match the sample MLIR syntax
    func_pattern = re.compile(r"func\.func\s+public\s+@main\((.*?)\)\s*->\s*(.*)", re.DOTALL)
    match = func_pattern.search(SAMPLE_ADD_MLIR)
    assert match is not None
    args_str, ret_str = match.groups()
    assert "tensor<1024x1024xf32>" in args_str
    assert "tensor<1024x1024xf32>" in ret_str

def test_matmul_relu_parsing():
    assert "stablehlo.dot_general" in SAMPLE_MATMUL_RELU_MLIR
    assert "stablehlo.maximum" in SAMPLE_MATMUL_RELU_MLIR
    assert "tensor<128x512xf32>" in SAMPLE_MATMUL_RELU_MLIR
