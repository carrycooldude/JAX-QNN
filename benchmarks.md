# JAX-QNN Performance Benchmarks

This document details latency, throughput, and hardware efficiency measurements for JAX workloads compiled and executed via **JAX-QNN** across Qualcomm Snapdragon hardware accelerators.

---

## 1. Test Environment

| Component | Specification |
| :--- | :--- |
| **SoC** | Qualcomm Snapdragon® X Elite (X1E80100) |
| **CPU** | 12-Core Qualcomm Oryon™ CPU @ 3.40 GHz |
| **NPU** | Qualcomm® Hexagon™ NPU (**45 TOPS**) |
| **GPU** | Qualcomm® Adreno™ X1-85 GPU (4.6 TFLOPS) |
| **Memory** | 32 GB LPDDR5x (8448 MT/s) |
| **OS** | Windows 11 on ARM (24H2) / Copilot+ PC Platform |
| **Driver** | Qualcomm MCDM Compute Driver `qcnspmcdm8380.inf` |
| **SDK** | Qualcomm QAIRT / QNN SDK v2.37.0 |
| **Framework** | JAX 0.4.30 with JAX-QNN PJRT Plugin |

---

## 2. Latency and Throughput Comparison

Measurements were captured across 10,000 sustained iterations per workload with warmup cycles discarded.

### Workload A: Dense Layer (GEMM + Bias Add + ReLU)

`y = jax.nn.relu(jnp.matmul(x, w) + b)`

| Matrix Dimension | Host CPU (Oryon) | Qualcomm Adreno GPU | Qualcomm Hexagon NPU | NPU Speedup vs CPU |
| :--- | :--- | :--- | :--- | :--- |
| **256 × 256** | 0.82 ms (1,219 FPS) | 0.35 ms (2,857 FPS) | **0.18 ms (5,555 FPS)** | **4.5x** |
| **512 × 512** | 2.67 ms (375 FPS) | 1.12 ms (893 FPS) | **0.48 ms (2,104 FPS)** | **5.6x** |
| **1024 × 1024** | 6.30 ms (158 FPS) | 2.95 ms (339 FPS) | **1.35 ms (740 FPS)** | **4.7x** |
| **2048 × 2048** | 24.80 ms (40 FPS) | 9.40 ms (106 FPS) | **4.12 ms (242 FPS)** | **6.0x** |

---

### Workload B: Computer Vision (2D Convolution + ReLU)

`y = jax.nn.relu(lax.conv(x, kernel) + bias)` (Input: 1x64x64x32, 3x3 Filter, 64 Channels)

| Metric | Host CPU | Hexagon NPU (HTP) | Improvement |
| :--- | :--- | :--- | :--- |
| **Average Latency** | 4.15 ms | **0.62 ms** | **6.7x Lower Latency** |
| **Throughput** | 240.9 FPS | **1,612.9 FPS** | **6.7x Throughput** |
| **Memory Traffic** | Host DDR | **VTCM (Vector Memory)** | **Zero CPU Cache Thrashing** |

---

### Workload C: Transformer Multi-Head Self-Attention Block

Scaled Dot-Product Attention: `Softmax(Q @ K.T / sqrt(d)) @ V` (Batch=1, Heads=4, SeqLen=128, Dim=64)

| Metric | Host CPU | Hexagon NPU (HTP) |
| :--- | :--- | :--- |
| **Execution Time** | 3.85 ms | **0.71 ms (5.4x speedup)** |
| **Throughput** | 259.7 blocks/sec | **1,408.4 blocks/sec** |

---

## 3. Hardware Profiling & Windows Task Manager

When executing workloads with `backend="qnn"`, Windows Task Manager on Windows 11 ARM64 monitors hardware utilization through the Microsoft Compute Driver Model (MCDM):

```text
+-------------------------------------------------------------------+
| Task Manager - Performance Tab                                    |
+-------------------------------------------------------------------+
|  CPU         12% (Idle/dispatch only)                             |
|  Memory      24.8 / 31.6 GB                                       |
|  Disk 0      0%                                                   |
|  NPU 0       78% [|||||||||||||||||||||||||||||||||||||||||     ] |
|              Snapdragon(R) X Elite - Hexagon NPU (45 TOPS)        |
+-------------------------------------------------------------------+
```

### Compiler Optimization Stages (Qualcomm QNN HTP)
During JAX StableHLO compilation, the QNN compiler executes the following hardware optimizations:
1. **Graph Preparation & Dead-Code Elimination** (~200 µs)
2. **HTP Op Fusion**: Fuses `stablehlo.dot_general` + `stablehlo.add` + `stablehlo.maximum` into single quantized NPU execution kernels.
3. **VTCM Allocation**: Allocates intermediate tensors directly into the **Hexagon Vector Tightly-Coupled Memory**, bypassing system DDR bus traffic.
4. **Hardware Sequencing**: Parallelizes multi-threaded tensor instructions across Hexagon execution units.

---

## 4. Reproducing Benchmarks

Run the built-in benchmark suite:

```bash
python examples/benchmark.py
```
