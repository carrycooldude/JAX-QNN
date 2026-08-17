"""
Example: Dense layer (MatMul + ReLU) on Qualcomm QNN.
"""

import jax
import jax.numpy as jnp
import jax_qnn

@jax.jit
def dense_layer(x, w, b):
    return jax.nn.relu(jnp.matmul(x, w) + b)

if __name__ == "__main__":
    x = jnp.ones((128, 256), dtype=jnp.float32)
    w = jnp.ones((256, 512), dtype=jnp.float32)
    b = jnp.zeros((512,), dtype=jnp.float32)

    out = jax.jit(dense_layer, backend="qnn")(x, w, b)
    
    # Verify correctness against standard execution
    expected = dense_layer(x, w, b)
    assert jnp.allclose(out, expected), "Mismatch between QNN and CPU reference!"
    
    print(f"Success: out.shape={out.shape}")
