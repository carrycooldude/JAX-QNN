# JAX-QNN: Qualcomm QNN Backend for JAX

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Target](https://img.shields.io/badge/Target-Snapdragon_NPU_%2F_GPU_%2F_CPU-red.svg)]()
[![Backend](https://img.shields.io/badge/JAX-PJRT_Plugin-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/Qualcomm-Hexagon_45_TOPS-purple.svg)]()
[![Docs](https://img.shields.io/badge/Docs-GitHub_Pages-0ea5e9.svg)](https://carrycooldude.github.io/JAX-QNN/)

**JAX-QNN** is an open-source native backend for **JAX** targeting Qualcomm Snapdragon hardware accelerators (**Hexagon NPU, Adreno GPU, and Oryon CPU**) via Qualcomm AI Engine Direct (QNN) and the OpenXLA **PJRT C API**.

It enables standard JAX code to compile StableHLO intermediate representations directly into native Qualcomm execution graphs with hardware acceleration on Windows on ARM, Linux, and Android.

---

## Performance Highlights (Snapdragon® X Elite)

Measured on a **12-Core Snapdragon® X Elite (X1E80100) with 45 TOPS Qualcomm® Hexagon™ NPU**:

| Workload | Host CPU | Qualcomm Adreno GPU | Qualcomm Hexagon NPU | Speedup vs CPU |
| :--- | :--- | :--- | :--- | :--- |
| **512×512 Dense Layer (GEMM + ReLU)** | 2.67 ms (375 FPS) | 1.12 ms (893 FPS) | **0.48 ms (2,104 FPS)** | **5.6x** |
| **1024×1024 Dense Layer** | 6.30 ms (158 FPS) | 2.95 ms (339 FPS) | **1.35 ms (740 FPS)** | **4.7x** |
| **2D Convolution (64x64x32, 3x3)** | 4.15 ms (241 FPS) | 1.45 ms (690 FPS) | **0.62 ms (1,613 FPS)** | **6.7x** |
| **Transformer Self-Attention (Seq 128)** | 3.85 ms (260 FPS) | 1.60 ms (625 FPS) | **0.71 ms (1,408 FPS)** | **5.4x** |

> See [docs/benchmarks.md](docs/benchmarks.md) for full benchmark methodology and memory bandwidth analysis.

---

## Hardware Targets

| Accelerator | QNN Engine | Best For | Typical Throughput |
| :--- | :--- | :--- | :--- |
| **Hexagon NPU (HTP)** | `QnnHtp.dll` / `libQnnHtp.so` | Low-latency inference, Quantized INT8/FP16, Transformers | Up to 2,100+ ops/sec |
| **Adreno GPU** | `QnnGpu.dll` / `libQnnGpu.so` | Floating-point FP32/FP16 matrix math & Computer Vision | Up to 890+ ops/sec |
| **Host CPU** | `QnnCpu.dll` / Reference | Development, fallback verification, and CPU debugging | Up to 375 ops/sec |

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
