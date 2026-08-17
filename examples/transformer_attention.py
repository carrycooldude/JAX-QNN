"""
Example: Scaled Dot-Product Attention on Qualcomm QNN.
"""

import jax
import jax.numpy as jnp
import jax_qnn

@jax.jit
def scaled_dot_product_attention(q, k, v):
    d_k = q.shape[-1]
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, v)

if __name__ == "__main__":
    batch, heads, seq_len, d_k = 1, 4, 128, 64
    q = jnp.ones((batch, heads, seq_len, d_k), dtype=jnp.float32)
    k = jnp.ones((batch, heads, seq_len, d_k), dtype=jnp.float32)
    v = jnp.ones((batch, heads, seq_len, d_k), dtype=jnp.float32)

    out = jax.jit(scaled_dot_product_attention, backend="qnn")(q, k, v)
    
    # Verify correctness against standard execution
    expected = scaled_dot_product_attention(q, k, v)
    assert jnp.allclose(out, expected, atol=1e-5), "Mismatch between QNN and reference!"
    
    print(f"Success: out.shape={out.shape}")
