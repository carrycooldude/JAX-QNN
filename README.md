# JAX-QNN: Qualcomm QNN Backend for JAX

[![PyPI version](https://img.shields.io/pypi/v/jax-qnn.svg?color=blue)](https://pypi.org/project/jax-qnn/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Target](https://img.shields.io/badge/Target-Snapdragon_NPU_%2F_GPU_%2F_CPU-red.svg)]()
[![Backend](https://img.shields.io/badge/JAX-PJRT_Plugin-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/Qualcomm-Hexagon_45_TOPS-purple.svg)]()
[![Docs](https://img.shields.io/badge/Docs-GitHub_Pages-0ea5e9.svg)](https://carrycooldude.github.io/JAX-QNN/)

**JAX-QNN** is an open-source native backend for **JAX** targeting Qualcomm Snapdragon hardware accelerators (**Hexagon NPU, Adreno GPU, and Oryon CPU**) via Qualcomm AI Engine Direct (QNN) and the OpenXLA **PJRT C API**.

It enables standard JAX code to compile StableHLO intermediate representations directly into native Qualcomm execution graphs with hardware acceleration on Windows on ARM, Linux, and Android.

---

## Installation

```bash
pip install jax-qnn
```

---

## Performance (Snapdragon® X Elite)

Measured on a **12-Core Snapdragon® X Elite (X1E80100)** using `python examples/benchmark.py`:

| Workload | CPU (Measured) | QNN Backend (Measured) | Speedup |
| :--- | :--- | :--- | :--- |
| **256×256 Dense GEMM + ReLU** | 0.38 ms | 0.42 ms | ~1.0x |
| **512×512 Dense GEMM + ReLU** | 1.83 ms | 1.61 ms | 1.1x |
| **1024×1024 Dense GEMM + ReLU** | 6.07 ms | 6.12 ms | ~1.0x |

> **Note:** Current QNN backend routes through the CPU reference runtime. When the full Hexagon NPU (HTP) hardware path is enabled via `QnnHtp.dll`, significantly higher speedups are expected due to VTCM on-chip memory and hardware op-fusion. Run `python examples/benchmark.py` to reproduce on your device.

> See [docs/benchmarks.md](docs/benchmarks.md) for full benchmark methodology.

---

## Hardware Targets

| Accelerator | QNN Engine | Best For |
| :--- | :--- | :--- |
| **Hexagon NPU (HTP)** | `QnnHtp.dll` / `libQnnHtp.so` | Low-latency inference, Quantized INT8/FP16, Transformers |
| **Adreno GPU** | `QnnGpu.dll` / `libQnnGpu.so` | Floating-point FP32/FP16 matrix math & Computer Vision |
| **Host CPU** | `QnnCpu.dll` / Reference | Development, fallback verification, and CPU debugging |

---

## Live Qualcomm Hardware Validation (NPU & GPU)

Run the hardware verification tool to test direct execution on both the **45 TOPS Hexagon NPU** and the **Adreno GPU**:

```bash
python tools/test_qualcomm_hardware.py
```

**Live Execution Log (Snapdragon® X Elite on Windows 11 ARM64):**

```text
Loading native library: C:\Users\rawat\JAX-QNN\build\pjrt_qnn.dll

=======================================================
 Testing Qualcomm Target: Qualcomm Hexagon NPU (HTP) (ID=1)
=======================================================
[JAX-QNN] QnnInterface successfully bound (providers: 1)
 <W> Initializing HtpProvider
[JAX-QNN] QnnBackend created successfully.
 <W> Logs will be sent to the system's default channel
 <E> DspTransport.openSession qnn_open failed, 0x80000406, prio 100
 <E> IDspTransport: Unable to load lib 0x80000406
 <E> DspTransport.getHandle failed, error 0x00000008
 <E> createDspTransportInstance failed to config transport object
 <E> error in creation of transport instance
 <W> Failed to create transport instance: 1002
 <W> Failed to load skel, error: 1002
 <W> Traditional path not available. Switching to user driver path
 <W> HTP user driver is loaded. Switched to user driver path
[JAX-QNN] QnnContext created successfully.
[JAX-QNN] Native QNN Hardware Backend Initialized: C:\Users\rawat\Downloads\v2.37.0.250724\qairt\2.37.0.250724\lib\aarch64-windows-msvc\QnnHtp.dll
[+] Init Result: 0 (Backend: C:\Users\rawat\Downloads\v2.37.0.250724\qairt\2.37.0.250724\lib\aarch64-windows-msvc\QnnHtp.dll)
[JAX-QNN] Parsing StableHLO (425 bytes)
[JAX-QNN] Parsed computation 'main': 2 inputs, 1 outputs, 3 ops
[JAX-QNN]   Input 0: [4, 4] f32
[JAX-QNN]   Input 1: [4, 4] f32
[JAX-QNN]   Op: stablehlo.dot_general (inputs: 2, outputs: 0)
[JAX-QNN]   Op: stablehlo.maximum (inputs: 2, outputs: 1)
[JAX-QNN]   Op: stablehlo.return (inputs: 1, outputs: 0)
[+] Graph Compiled Successfully (Handle: 0x243694f18c0)
[+] Inputs: 2, Outputs: 1
[+] Hardware Execution Status: 0
[+] Correctness Verified! Output[0, 0] = 24.0 (Expected: 24.0)

=======================================================
 Testing Qualcomm Target: Qualcomm Adreno GPU (OpenCL/Direct3D) (ID=2)
=======================================================
 <W> Logs will be sent to the system's default channel
 <W> m_CFBCallbackInfoObj is not initialized, return emptyList
 <W> Logs will be sent to the system's default channel
 <W> Logs will be sent to the system's default channel
[JAX-QNN] QnnInterface successfully bound (providers: 1)
[JAX-QNN] QnnBackend created successfully.
[JAX-QNN] QnnContext created successfully.
[JAX-QNN] Native QNN Hardware Backend Initialized: C:\Users\rawat\Downloads\v2.37.0.250724\qairt\2.37.0.250724\lib\aarch64-windows-msvc\QnnGpu.dll
[+] Init Result: 0 (Backend: C:\Users\rawat\Downloads\v2.37.0.250724\qairt\2.37.0.250724\lib\aarch64-windows-msvc\QnnGpu.dll)
[JAX-QNN] Parsing StableHLO (425 bytes)
[JAX-QNN] Parsed computation 'main': 2 inputs, 1 outputs, 3 ops
[JAX-QNN]   Input 0: [4, 4] f32
[JAX-QNN]   Input 1: [4, 4] f32
[JAX-QNN]   Op: stablehlo.dot_general (inputs: 2, outputs: 0)
[JAX-QNN]   Op: stablehlo.maximum (inputs: 2, outputs: 1)
[JAX-QNN]   Op: stablehlo.return (inputs: 1, outputs: 0)
[+] Graph Compiled Successfully (Handle: 0x243800dc6f0)
[+] Inputs: 2, Outputs: 1
[+] Hardware Execution Status: 0
[+] Correctness Verified! Output[0, 0] = 24.0 (Expected: 24.0)
```

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/carrycooldude/JAX-QNN.git
cd JAX-QNN

# Install in editable mode
pip install -e .
```

### 2. Basic Example

```python
import jax
import jax.numpy as jnp
import jax_qnn

# 1. Discover devices
print("Devices:", jax.devices("qnn"))

# 2. Define standard JAX model
@jax.jit
def model(x, w, b):
    return jax.nn.relu(jnp.matmul(x, w) + b)

# 3. Create inputs
x = jnp.ones((128, 512), dtype=jnp.float32)
w = jnp.ones((512, 512), dtype=jnp.float32)
b = jnp.zeros((512,), dtype=jnp.float32)

# 4. Execute on Qualcomm Hardware
output = jax.jit(model, backend="qnn")(x, w, b)
print("Output shape:", output.shape)
```

---

## Available Examples

| Example Script | Description | Hardware Targets |
| :--- | :--- | :--- |
| [`examples/simple_add.py`](examples/simple_add.py) | Minimal elementwise tensor addition | NPU / GPU / CPU |
| [`examples/matmul_relu.py`](examples/matmul_relu.py) | Dense linear layer with ReLU activation | NPU / GPU / CPU |
| [`examples/conv2d_relu.py`](examples/conv2d_relu.py) | 2D image convolution with bias and activation | NPU / GPU / CPU |
| [`examples/transformer_attention.py`](examples/transformer_attention.py) | Scaled dot-product Multi-Head Self Attention | NPU / GPU / CPU |
| [`examples/benchmark.py`](examples/benchmark.py) | Multi-hardware latency & FPS benchmarking suite | NPU vs GPU vs CPU |

Run any example directly:
```bash
python examples/matmul_relu.py
python examples/conv2d_relu.py
python examples/transformer_attention.py
python examples/benchmark.py
```

---

## Contributing Examples

We actively encourage community contributions of new JAX models and benchmarks!

Whether you want to add:
- **Vision Models** (ResNet, MobileNet, ViT)
- **Language Models** (Llama, Mistral decoder blocks, RoPE, RMSNorm)
- **Audio Models** (Whisper encoder, Conformer blocks)
- **Quantization Examples** (INT8, FP16 GEMM)

Please check our [Contributing Examples Guide](docs/examples_guide.md) to get started!

---

## Architecture Overview

```text
                  Python / JAX Code
                         │
           jax.jit(..., backend="qnn")
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
             csrc/stablehlo_to_qnn.cc
                         │
       Direct Qualcomm Native QNN C API:
         - QnnBackend_create()
         - QnnContext_create()
         - QnnGraph_create()
         - QnnGraph_addNode()
         - QnnGraph_finalize()
         - QnnGraph_execute()
                         │
           Qualcomm Hexagon NPU / HTP Hardware
```

For comprehensive technical specifications, refer to [docs/architecture.md](docs/architecture.md) and [docs/setup.md](docs/setup.md).

---

## License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
