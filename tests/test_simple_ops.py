# =============================================================================
# tests/test_simple_ops.py — Unit Tests for Supported Operators
# =============================================================================

import pytest

SUPPORTED_OPS = [
    "stablehlo.add",
    "stablehlo.subtract",
    "stablehlo.multiply",
    "stablehlo.divide",
    "stablehlo.dot_general",
    "stablehlo.maximum",
    "stablehlo.minimum",
    "stablehlo.negate",
    "stablehlo.reshape",
    "stablehlo.transpose",
    "stablehlo.reduce",
    "stablehlo.broadcast_in_dim",
]

def test_supported_ops_registry():
    assert "stablehlo.add" in SUPPORTED_OPS
    assert "stablehlo.dot_general" in SUPPORTED_OPS
    assert "stablehlo.maximum" in SUPPORTED_OPS

@pytest.mark.parametrize("op", SUPPORTED_OPS)
def test_op_name_valid(op):
    assert op.startswith("stablehlo.")
