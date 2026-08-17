"""
Example: 2D Convolution + ReLU on Qualcomm QNN.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax_qnn

@jax.jit
def conv_layer(x, kernel, bias):
    conv_out = lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return jax.nn.relu(conv_out + bias)

if __name__ == "__main__":
    N, H, W, C_in, C_out = 1, 64, 64, 32, 64
    x = jnp.ones((N, H, W, C_in), dtype=jnp.float32)
    kernel = jnp.ones((3, 3, C_in, C_out), dtype=jnp.float32) / 9.0
    bias = jnp.zeros((C_out,), dtype=jnp.float32)

    out = jax.jit(conv_layer, backend="qnn")(x, kernel, bias)
    
    # Verify correctness against standard execution
    expected = conv_layer(x, kernel, bias)
    assert jnp.allclose(out, expected, atol=1e-5), "Mismatch between QNN and reference!"
    
    print(f"Success: out.shape={out.shape}")
