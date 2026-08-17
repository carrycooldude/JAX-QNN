# =============================================================================
# examples/simple_add.py — Minimal JAX-QNN Example
# =============================================================================

"""
Minimal example demonstrating elementwise addition compiled and executed
via the Qualcomm QNN backend for JAX.
"""

import jax
import jax.numpy as jnp
import jax_qnn


def main():
    print("=== JAX QNN Backend: Simple Add ===")
    print(f"Available QNN devices: {jax.devices('qnn')}")

    @jax.jit
    def add_fn(x, y):
        return x + y

    x = jnp.ones((1024, 1024), dtype=jnp.float32)
    y = jnp.ones((1024, 1024), dtype=jnp.float32)

    result = jax.jit(add_fn, backend="qnn")(x, y)
    print(f"Result shape: {result.shape}")
    print(f"Result sum: {float(jnp.sum(result))}")
    print(f"Expected sum: {1024 * 1024 * 2.0}")


if __name__ == "__main__":
    main()
