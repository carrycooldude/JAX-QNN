# =============================================================================
# jax_qnn.qnn_config — Configuration and Hardware Utilities
# =============================================================================

import os
from typing import Optional

def set_qnn_sdk_root(path: str) -> None:
    """Sets the QNN_SDK_ROOT environment variable."""
    os.environ["QNN_SDK_ROOT"] = os.path.abspath(path)

def get_qnn_sdk_root() -> Optional[str]:
    """Returns the QNN_SDK_ROOT path if set."""
    return os.environ.get("QNN_SDK_ROOT")

def is_qnn_sdk_available() -> bool:
    """Checks whether the Qualcomm QNN SDK headers and binaries exist."""
    root = get_qnn_sdk_root()
    if not root or not os.path.isdir(root):
        return False
    
    header_check = os.path.join(root, "include", "QNN", "QnnInterface.h")
    return os.path.exists(header_check)

def get_target_device_info() -> dict:
    """Returns device properties for Qualcomm target."""
    return {
        "platform": "qnn",
        "has_qnn_sdk": is_qnn_sdk_available(),
        "qnn_sdk_root": get_qnn_sdk_root(),
    }
