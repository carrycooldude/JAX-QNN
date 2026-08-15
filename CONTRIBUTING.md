# Contributing to JAX-QNN

Thank you for your interest in contributing to **JAX-QNN**! We welcome contributions ranging from new operator lowerings and performance optimizations to documentation and runnable examples.

---

## 🛠️ Development Setup

### 1. Prerequisites
- **Python 3.10+** (with `pip` and virtual environment support)
- **CMake >= 3.20**
- **C++17 Compiler** (MSVC 2022 on Windows, Clang on Linux/Android)
- **Qualcomm QNN / QAIRT SDK** (optional, STUB mode is enabled if not present)

### 2. Quickstart
```bash
# Clone the repository
git clone https://github.com/carrycooldude/JAX-QNN.git
cd JAX-QNN

# Install Python package in editable mode with development dependencies
pip install -e ".[dev]"

# Build the C++ PJRT plugin
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

---

## 🧪 Running Tests

We maintain a 100% passing test suite across operator matrices, compilation signatures, and discovery routines:

```bash
# Run the test suite
pytest tests/ -v
```

---

## 📁 Repository Organization

```text
JAX-QNN/
├── CMakeLists.txt           # Build system for C++ PJRT plugin
├── pyproject.toml           # Python packaging with jax_plugins entry point
├── csrc/                    # Native C++ PJRT plugin source
│   ├── pjrt/pjrt_c_api.h    # OpenXLA PJRT C API definitions
│   ├── pjrt_qnn_plugin.cc   # GetPjRtApi dispatch table
│   ├── qnn_client.cc        # Client lifecycle & device enumeration
│   ├── qnn_device.cc        # Qualcomm device & memory descriptors
│   ├── qnn_executable.cc    # Graph execution engine
│   ├── qnn_buffer.cc        # Host <-> Device tensor buffer management
│   ├── stablehlo_to_qnn.cc  # StableHLO / MLIR parser and translator
│   └── qnn_runtime.cc       # Dynamic Qualcomm QNN SDK loader
├── jax_plugins/qnn/         # JAX auto-discovery plugin namespace
├── jax_qnn/                 # Public Python configuration API
├── examples/                # Runnable examples (Vision, Attention, Dense)
├── tools/                   # Hardware & environment diagnostic tools
├── docs/                    # Architecture, setup, and benchmark docs
└── tests/                   # Pytest test suite
```

---

## 🚀 Submitting Contributions

1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make your changes**: Ensure formatting and comments are clean.
3. **Add tests / examples**: When adding new ops or models, add corresponding unit tests or example scripts.
4. **Verify tests pass**: Run `pytest tests/ -v`.
5. **Submit a Pull Request**: Provide a clear description of your changes and benchmark/test output.
