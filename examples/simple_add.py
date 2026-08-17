"""
Minimal example: Elementwise addition using JAX on Qualcomm QNN.
"""

import jax
import jax.numpy as jnp
import jax_qnn

@jax.jit
def add_fn(x, y):
    return x + y

if __name__ == "__main__":
    x = jnp.ones((1024, 1024), dtype=jnp.float32)
    y = jnp.ones((1024, 1024), dtype=jnp.float32)

    z = jax.jit(add_fn, backend="qnn")(x, y)
    
    # Verify correctness against standard execution
    expected = add_fn(x, y)
    assert jnp.allclose(z, expected), "Mismatch between QNN and CPU reference!"
    
    print(f"Success: z.shape={z.shape}, sum={float(jnp.sum(z))}")
