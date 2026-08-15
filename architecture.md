# JAX-QNN Architecture Specification

## 1. System Overview

```text
                  Python / JAX Code
                         │
                      jax.jit
                         │
                 JAX Core Tracing
                         │
                       JAXPR
                         │
                  MLIR Lowering
                         │
                     StableHLO
                         │
            PJRT_Client_Compile (C API)
                         │
             StableHLO → QNN Lowering
                         │
                  QNN Graph (Native)
                         │
                  QnnGraph_finalize
                         │
               QNN PJRT Executable
                         │
             PJRT_LoadedExecutable_Execute
                         │
                 QnnGraph_execute
                         │
           Qualcomm Hexagon NPU / HTP Runtime
```

## 2. Integration Boundary

JAX 0.11.0 discovers custom backends via the **PJRT C API** (`pjrt_c_api.h`):

1. **Discovery:**
   - JAX scans `jax_plugins` namespace modules and entry points defined in `pyproject.toml`.
   - `jax_plugins.qnn.initialize()` calls `jax._src.xla_bridge.register_plugin("qnn", library_path=...)`.
   - JAX dynamically loads the shared library (`pjrt_qnn.dll` / `pjrt_qnn.so`) and looks up `GetPjRtApi()`.

2. **Client Creation & Device Enumeration:**
   - `PJRT_Client_Create` initializes the internal `QnnClientState`.
   - `PJRT_Client_Devices` returns the addressable `QnnDeviceState` instances (e.g. Hexagon NPU device).

3. **Compilation Pipeline:**
   - JAX passes the serialized StableHLO program (MLIR text) into `PJRT_Client_Compile`.
   - The parser inspects the computation graph, types, and tensors.
   - Nodes are mapped 1-to-1 to QNN native op definitions (`QnnGraph_addNode`).
   - The graph is finalized via `QnnGraph_finalize`.

4. **Execution & Memory:**
   - Inputs are passed via `PJRT_Buffer` pointers.
   - Host-to-device transfers utilize Qualcomm unified memory where possible.
   - `QnnGraph_execute` runs the workload directly on the NPU.
   - Results are converted to host `PJRT_Buffer` arrays and returned to JAX asynchronously or synchronously.

## 3. Supported Op Matrix (MVP)

| StableHLO Primitive | QNN Operation | NPU Acceleration |
| :--- | :--- | :--- |
| `stablehlo.add` | `QNN_OP_ELEMENT_WISE_ADD` | ✅ Supported |
| `stablehlo.subtract` | `QNN_OP_ELEMENT_WISE_SUBTRACT` | ✅ Supported |
| `stablehlo.multiply` | `QNN_OP_ELEMENT_WISE_MULTIPLY` | ✅ Supported |
| `stablehlo.divide` | `QNN_OP_ELEMENT_WISE_DIVIDE` | ✅ Supported |
| `stablehlo.dot_general` | `QNN_OP_MAT_MUL` / `QNN_OP_FULLY_CONNECTED` | ✅ Supported |
| `stablehlo.maximum` | `QNN_OP_RELU` / `QNN_OP_ELEMENT_WISE_MAX` | ✅ Supported |
| `stablehlo.minimum` | `QNN_OP_ELEMENT_WISE_MIN` | ✅ Supported |
| `stablehlo.negate` | `QNN_OP_ELEMENT_WISE_NEG` | ✅ Supported |
| `stablehlo.reshape` | `QNN_OP_RESHAPE` | ✅ Supported |
| `stablehlo.transpose` | `QNN_OP_TRANSPOSE` | ✅ Supported |
| `stablehlo.reduce` | `QNN_OP_REDUCE_SUM` / `QNN_OP_REDUCE_MEAN` | ✅ Supported |
| `stablehlo.broadcast_in_dim`| `QNN_OP_TILE` / Broadcast View | ✅ Supported |
