# =============================================================================
# examples/simple_add.py — Minimal JAX-QNN Example
# =============================================================================

"""
Minimal example demonstrating elementwise addition compiled and executed
via the Qualcomm QNN backend for JAX.
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

# 1. Discover devices
print("Available devices:", jax.devices("qnn"))

# 2. Define computation
@jax.jit
def add_fn(x, y):
    return x + y

# 3. Create input tensors on device
x = jnp.ones((1024, 1024), dtype=jnp.float32)
y = jnp.ones((1024, 1024), dtype=jnp.float32)

# 4. Execute on Qualcomm hardware
z = jax.jit(add_fn, backend="qnn")(x, y)
print("Result shape:", z.shape)
print("Result preview:", z[0, :5])
''')
    sys.exit(0)

def main():
    print("=== JAX QNN Backend: Simple Add ===")
    print("Available devices:", jax.devices("qnn"))

    @jax.jit
    def add_fn(x, y):
        return x + y

    x = jnp.ones((1024, 1024), dtype=jnp.float32)
    y = jnp.ones((1024, 1024), dtype=jnp.float32)

    result = jax.jit(add_fn, backend="qnn")(x, y)
    print("Result shape:", result.shape)
    print("Result sum:", float(jnp.sum(result)))

if __name__ == "__main__":
    main()
