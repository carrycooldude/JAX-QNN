# =============================================================================
# examples/matmul_relu.py — Matrix Multiplication + Activation Example
# =============================================================================

"""
Demonstrates MatMul + ReLU fusion executed on Qualcomm NPU / QNN backend.
"""

import jax
import jax.numpy as jnp
import jax_qnn


def main():
    print("=== JAX QNN Backend: MatMul + ReLU ===")
    print(f"Available QNN devices: {jax.devices('qnn')}")

    @jax.jit
    def dense_layer(x, w, b):
        return jax.nn.relu(jnp.matmul(x, w) + b)

    x = jnp.ones((128, 256), dtype=jnp.float32)
    w = jnp.ones((256, 512), dtype=jnp.float32)
    b = jnp.zeros((512,), dtype=jnp.float32)

    y = jax.jit(dense_layer, backend="qnn")(x, w, b)
    print(f"Output shape: {y.shape}")
    print(f"Output sample [0, :5]: {y[0, :5]}")
    print(f"Output min: {float(jnp.min(y))}, max: {float(jnp.max(y))}")


if __name__ == "__main__":
    main()
