# =============================================================================
# tests/test_discovery.py — Backend Discovery Tests
# =============================================================================

import pytest

def test_config_sdk_root():
    from jax_qnn import set_qnn_sdk_root, get_qnn_sdk_root, is_qnn_sdk_available
    
    set_qnn_sdk_root("/dummy/qnn/sdk")
    assert get_qnn_sdk_root() is not None
    assert is_qnn_sdk_available() is False

def test_device_info():
    from jax_qnn import get_target_device_info
    info = get_target_device_info()
    assert info["platform"] == "qnn"
    assert "has_qnn_sdk" in info
