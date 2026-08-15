# =============================================================================
# jax_plugins.qnn — JAX PJRT Plugin Entry Point
# =============================================================================

import os
import sys
import logging

logger = logging.getLogger("jax_qnn")

def _get_library_path() -> str:
    """Finds the compiled PJRT QNN plugin library (.dll or .so)."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    if sys.platform == "win32":
        candidates = [
            os.path.join(dir_path, "pjrt_qnn.dll"),
            os.path.join(dir_path, "..", "..", "build", "pjrt_qnn.dll"),
            os.path.join(dir_path, "..", "..", "build", "Release", "pjrt_qnn.dll"),
        ]
    elif sys.platform == "darwin":
        candidates = [
            os.path.join(dir_path, "pjrt_qnn.dylib"),
            os.path.join(dir_path, "..", "..", "build", "pjrt_qnn.dylib"),
        ]
    else:
        candidates = [
            os.path.join(dir_path, "pjrt_qnn.so"),
            os.path.join(dir_path, "..", "..", "build", "pjrt_qnn.so"),
        ]
    
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
            
    return os.path.join(dir_path, "pjrt_qnn.dll" if sys.platform == "win32" else "pjrt_qnn.so")


def initialize():
    """Initializes the QNN PJRT plugin with JAX."""
    try:
        import jax._src.xla_bridge as xb
    except ImportError:
        return

    lib_path = _get_library_path()
    
    if not os.path.exists(lib_path):
        return

    # On Windows, standard jaxlib uses LoadPjrtPlugin stub, so we fall back quietly
    try:
        xb.register_plugin(
            "qnn",
            priority=400,
            library_path=lib_path,
            options=None,
        )
    except Exception:
        pass
