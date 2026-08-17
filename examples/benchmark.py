# =============================================================================
# examples/benchmark.py — Comprehensive JAX-QNN Multi-Hardware Benchmark
# =============================================================================

"""
Benchmarks execution latency and throughput across compute targets:
1. Qualcomm Hexagon NPU (HTP Backend)
2. Qualcomm Adreno GPU (QNN GPU Backend)
3. Reference CPU Backend
"""

import time
import sys
import jax
import jax.numpy as jnp
import jax_qnn

def run_benchmarks():
    print("=" * 70)
    print("   JAX-QNN: MULTI-HARDWARE LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 70)
    print(f"Available QNN devices: {jax.devices('qnn')}")
    print("Target Workloads: Dense Layer GEMM + ReLU (FP32)")
    print("-" * 70)

    test_shapes = [
        ("Small GEMM", 256, 256, 256),
        ("Medium GEMM", 512, 512, 512),
        ("Large GEMM", 1024, 1024, 1024),
    ]

    results = []

    for label, M, K, N in test_shapes:
        print(f"\n[*] Benchmarking {label} [{M}x{K} @ {K}x{N}]...")

        @jax.jit
        def model(x, w, b):
            return jax.nn.relu(jnp.matmul(x, w) + b)

        x = jnp.ones((M, K), dtype=jnp.float32)
        w = jnp.ones((K, N), dtype=jnp.float32)
        b = jnp.zeros((N,), dtype=jnp.float32)

        t_cpu = None
        t_qnn = None

        # 1. CPU Reference Execution
        try:
            _ = model(x, w, b).block_until_ready()  # warmup

            num_runs = 50
            t0 = time.perf_counter()
            for _ in range(num_runs):
                _ = model(x, w, b).block_until_ready()
            t_cpu = (time.perf_counter() - t0) / num_runs * 1000
            print(f"  - Host CPU           : {t_cpu:7.2f} ms | {1000.0 / t_cpu:8.1f} runs/sec")
        except Exception as e:
            print(f"  - Host CPU           : error ({e})")

        # 2. QNN Backend Execution
        try:
            qnn_model = jax.jit(model, backend="qnn")
            _ = qnn_model(x, w, b).block_until_ready()  # warmup

            num_runs = 100
            t0 = time.perf_counter()
            for _ in range(num_runs):
                _ = qnn_model(x, w, b).block_until_ready()
            t_qnn = (time.perf_counter() - t0) / num_runs * 1000
            speedup = f"{t_cpu / t_qnn:.1f}x" if t_cpu else "N/A"
            print(f"  - QNN Backend        : {t_qnn:7.2f} ms | {1000.0 / t_qnn:8.1f} runs/sec | Speedup: {speedup}")
        except Exception as e:
            print(f"  - QNN Backend        : error ({e})")

        results.append((label, f"{M}x{N}", t_cpu, t_qnn))

    # Print summary from actual measured data
    print("\n" + "=" * 70)
    print("   MEASURED RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Workload':<20} | {'Shape':<10} | {'CPU (ms)':>10} | {'QNN (ms)':>10} | {'Speedup':>8}")
    print("-" * 70)
    for label, shape, t_cpu, t_qnn in results:
        cpu_str = f"{t_cpu:.2f}" if t_cpu else "N/A"
        qnn_str = f"{t_qnn:.2f}" if t_qnn else "N/A"
        if t_cpu and t_qnn:
            speedup_str = f"{t_cpu / t_qnn:.1f}x"
        else:
            speedup_str = "N/A"
        print(f"{label:<20} | {shape:<10} | {cpu_str:>10} | {qnn_str:>10} | {speedup_str:>8}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
