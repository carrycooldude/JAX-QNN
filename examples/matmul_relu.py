# =============================================================================
# examples/matmul_relu.py — Matrix Multiplication + Activation Example
# =============================================================================

"""
Demonstrates MatMul + ReLU fusion executed on Qualcomm NPU / QNN backend.
"""

import sys

try:
    import jax
    import jax.numpy as jnp
    import jax_qnn
except ImportError as e:
    print(f"Note: Running demonstration mock due to missing dependencies: {e}")
    print("\nDemonstration code:")
    print('''
import jax
import jax.numpy as jnp

@jax.jit
def dense_layer(x, w, b):
    return jax.nn.relu(jnp.matmul(x, w) + b)

x = jnp.ones((128, 256), dtype=jnp.float32)
w = jnp.ones((256, 512), dtype=jnp.float32)
b = jnp.zeros((512,), dtype=jnp.float32)

y = jax.jit(dense_layer, backend="qnn")(x, w, b)
print("Output shape:", y.shape)
''')
    sys.exit(0)

def main():
    print("=== JAX QNN Backend: MatMul + ReLU ===")
    
    @jax.jit
    def dense_layer(x, w, b):
        return jax.nn.relu(jnp.matmul(x, w) + b)

    x = jnp.ones((128, 256), dtype=jnp.float32)
    w = jnp.ones((256, 512), dtype=jnp.float32)
    b = jnp.zeros((512,), dtype=jnp.float32)

    y = jax.jit(dense_layer, backend="qnn")(x, w, b)
    print("Output shape:", y.shape)
    print("Execution complete.")

if __name__ == "__main__":
    main()
