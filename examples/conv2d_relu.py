# =============================================================================
# examples/conv2d_relu.py — 2D Convolution + ReLU on QNN Backend
# =============================================================================

"""
Demonstrates a 2D Convolutional layer with ReLU activation targeting
the Qualcomm QNN backend (Hexagon NPU / Adreno GPU / CPU).
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax_qnn

print("=" * 65)
print("   JAX-QNN: 2D CONVOLUTION + RELU ON QUALCOMM HARDWARE")
print("=" * 65)

@jax.jit
def conv_layer(x, kernel, bias):
    # Standard Conv2D: input [N, H, W, C], kernel [H_k, W_k, C_in, C_out]
    conv_out = lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    return jax.nn.relu(conv_out + bias)

# Input image tensor: Batch=1, Height=64, Width=64, Channels=32
N, H, W, C_in, C_out = 1, 64, 64, 32, 64
x = jnp.ones((N, H, W, C_in), dtype=jnp.float32)

# Kernel: 3x3 filter
kernel = jnp.ones((3, 3, C_in, C_out), dtype=jnp.float32) / 9.0
bias = jnp.zeros((C_out,), dtype=jnp.float32)

print(f"[*] Input shape: {x.shape}")
print(f"[*] Kernel shape: {kernel.shape}")

# Execute on QNN backend
out = jax.jit(conv_layer, backend="qnn")(x, kernel, bias)

print(f"[+] Output shape: {out.shape}")
print(f"[+] Output sample (first 4 channels of pixel 0,0): {out[0, 0, 0, :4]}")
print("=" * 65)
