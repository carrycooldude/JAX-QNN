# =============================================================================
# jax_qnn — Qualcomm QNN Backend for JAX
# =============================================================================

"""
JAX-QNN: Execute JAX computations on Qualcomm Snapdragon NPU / Hexagon DSP via QNN.
"""

import os
import sys
import logging
import numpy as np

logger = logging.getLogger("jax_qnn")

from jax_qnn.bridge import (
    is_native_available,
    get_backend_name,
    compile_stablehlo,
    NativeQnnExecutable,
)

class QnnLoadedExecutable:
    """Wrapper that executes a compiled native QNN graph on Qualcomm hardware."""

    def __init__(self, native_exec: NativeQnnExecutable, base_executable, client):
        self._native_exec = native_exec
        self._base_executable = base_executable
        self._client = client

    def execute(self, arguments):
        """Executes computation with input buffers and returns output buffers."""
        try:
            # Extract numpy arrays from JAX buffer arguments
            np_args = [np.asarray(arg) for arg in arguments]
            outputs = self._native_exec(*np_args)
            
            # Wrap output numpy arrays back into JAX device buffers
            device = self._client.devices()[0]
            return [self._client.buffer_from_pyval(out, device) for out in outputs]
        except Exception as e:
            logger.warning("Native QNN execution fallback to base executor: %s", e)
            return self._base_executable.execute(arguments)

    def execute_sharded(self, arguments):
        return self._base_executable.execute_sharded(arguments)

    def __getattr__(self, name):
        return getattr(self._base_executable, name)


def _register_qnn_backend():
    try:
        import jax._src.xla_bridge as xb
        import jaxlib.xla_client as xc

        class QnnClient:
            def __init__(self):
                self._base_client = xc.make_cpu_client()
                self.platform = "qnn"
                self.platform_version = f"0.1.0 (Qualcomm AI Engine Direct / {get_backend_name()})"
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
                base_exec = self._base_client.compile(computation, compile_options)

                # Try native QNN compilation if DLL and MLIR are available
                if is_native_available():
                    try:
                        # Extract MLIR / StableHLO text from JAX computation
                        mlir_str = None
                        if hasattr(computation, "as_mlir_module"):
                            mlir_str = str(computation.as_mlir_module())
                        elif hasattr(computation, "as_hlo_text"):
                            mlir_str = str(computation.as_hlo_text())
                        elif hasattr(computation, "as_serialized_hlo_module_proto"):
                            # If binary proto, we can also extract text
                            mlir_str = str(computation)

                        if mlir_str and ("stablehlo." in mlir_str or "func.func" in mlir_str):
                            native_exec = compile_stablehlo(mlir_str)
                            return QnnLoadedExecutable(native_exec, base_exec, self)
                    except Exception as e:
                        logger.debug("Native QNN compilation fallback: %s", e)

                return base_exec

            def buffer_from_pyval(self, val, device=None):
                if device is None:
                    device = self._devices[0]
                return self._base_client.buffer_from_pyval(val, device)

            def __getattr__(self, name):
                return getattr(self._base_client, name)

        if "qnn" not in xb._backend_factories:
            xb.register_backend_factory("qnn", lambda: QnnClient(), priority=500, fail_quietly=False)
            logger.info("Successfully registered 'qnn' backend in JAX with native QNN bridge.")
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
    "is_native_available",
    "get_backend_name",
]
