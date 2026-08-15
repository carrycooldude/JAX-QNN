# =============================================================================
# examples/transformer_attention.py — Multi-Head Self Attention on QNN
# =============================================================================

"""
Demonstrates a scaled dot-product Multi-Head Self-Attention block
compiled and executed via JAX-QNN on Qualcomm NPU / GPU / CPU.
"""

import jax
import jax.numpy as jnp
import jax_qnn

print("=" * 65)
print("   JAX-QNN: MULTI-HEAD SELF-ATTENTION ON QUALCOMM HARDWARE")
print("=" * 65)

@jax.jit
def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: [batch_size, num_heads, seq_len, d_k]
    d_k = q.shape[-1]
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask
        
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, v)

# Problem dimensions: Batch=1, Heads=4, SeqLen=128, DimPerHead=64
batch, heads, seq_len, d_k = 1, 4, 128, 64
q = jnp.ones((batch, heads, seq_len, d_k), dtype=jnp.float32)
k = jnp.ones((batch, heads, seq_len, d_k), dtype=jnp.float32)
v = jnp.ones((batch, heads, seq_len, d_k), dtype=jnp.float32)

print(f"[*] Query/Key/Value shape: {q.shape}")

# Execute on QNN backend
output = jax.jit(scaled_dot_product_attention, backend="qnn")(q, k, v)

print(f"[+] Attention Output shape: {output.shape}")
print(f"[+] Mean output value: {float(jnp.mean(output)):.4f}")
print("=" * 65)
