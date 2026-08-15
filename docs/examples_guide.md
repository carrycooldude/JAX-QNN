# Contributing Examples to JAX-QNN

We welcome community contributions of new JAX models, neural network architectures, and benchmarks running on Qualcomm hardware via **JAX-QNN**.

---

## 1. Example Structure

All examples belong in the [`examples/`](file:///c:/Users/rawat/JAX-QNN/examples) directory. A clean example should:
1. Be self-contained and runnable directly with `python examples/your_example.py`.
2. Use standard `jax.numpy` and `@jax.jit(..., backend="qnn")`.
3. Provide clear printouts of input shapes, output shapes, and verification results.

---

## 2. Template for a New Example

```python
import jax
import jax.numpy as jnp
import jax_qnn

def main():
    print("=== JAX-QNN Example: [Your Model Name] ===")
    
    # 1. Define Model Architecture
    @jax.jit
    def my_model(x, params):
        # Your JAX computation here
        return jnp.matmul(x, params['w']) + params['b']

    # 2. Prepare Inputs
    x = jnp.ones((32, 128), dtype=jnp.float32)
    params = {
        'w': jnp.ones((128, 64), dtype=jnp.float32),
        'b': jnp.zeros((64,), dtype=jnp.float32),
    }

    # 3. Execute on QNN Target (NPU / GPU / CPU)
    output = jax.jit(my_model, backend="qnn")(x, params)

    print("[+] Execution finished successfully!")
    print(f"[+] Output shape: {output.shape}")

if __name__ == "__main__":
    main()
```

---

## 3. High-Priority Areas for New Examples

We are actively seeking contributions in the following domains:

- **Large Language Models (LLMs)**:
  - RMSNorm, RoPE (Rotary Position Embeddings), SwiGLU activations.
  - Llama / Mistral decoder layers in pure JAX.
- **Vision Models**:
  - ResNet residual blocks, MobileNet depthwise separable convolutions.
  - Vision Transformer (ViT) patch embedding and attention blocks.
- **Audio & Speech**:
  - 1D Convolutions, STFT / Mel-Spectrogram feature extractors.
  - Conformer / Whisper encoder blocks.
- **Quantization & Mixed Precision**:
  - FP16, BF16, and INT8 INT4 quantized GEMM examples.

---

## 4. How to Submit

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/example-my-model
   ```
2. Add your example script to `examples/`.
3. Add a unit test in `tests/` if you are verifying numerical correctness against CPU JAX.
4. Open a Pull Request with a short description and execution logs.
