# Building and Installing JAX-QNN

This guide walks you through configuring the Qualcomm QNN SDK, compiling the PJRT plugin C++ library, and installing the `jax-qnn` Python package.

---

## 1. Prerequisites

1. **Python 3.10+** (with development headers)
2. **CMake >= 3.20**
3. **C++17 Compiler**:
   - On Windows: Visual Studio 2022 / MSVC (`cl.exe`) or Clang
   - On Linux / Android: Clang 14+ or GCC 11+
4. **Qualcomm AI Runtime (QAIRT / QNN SDK)**:
   - Download from [Qualcomm Software Center](https://softwarecenter.qualcomm.com/) or Qualcomm Package Manager (QPM).
   - Expected directory structure:
     ```text
     $QNN_SDK_ROOT/
     ├── include/
     │   └── QNN/
     │       ├── QnnInterface.h
     │       ├── QnnBackend.h
     │       ├── QnnContext.h
     │       ├── QnnGraph.h
     │       └── ...
     └── lib/
         ├── aarch64-windows-msvc/   (Windows ARM64)
         │   ├── QnnHtp.dll
         │   └── QnnCpu.dll
         └── aarch64-linux-clang/    (Linux / Android)
             ├── libQnnHtp.so
             └── libQnnCpu.so
     ```

---

## 2. Setting Environment Variables

Set the `QNN_SDK_ROOT` environment variable to point to your installed SDK:

### On Windows PowerShell:
```powershell
$env:QNN_SDK_ROOT = "C:\Qualcomm\QAIRT\2.47.0.260601"
```

### On Linux / macOS / Bash:
```bash
export QNN_SDK_ROOT="/opt/qualcomm/qairt/2.47.0.260601"
```

---

## 3. Building the C++ PJRT Plugin

```bash
# 1. Create build directory
mkdir build
cd build

# 2. Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DQNN_SDK_ROOT=$QNN_SDK_ROOT

# 3. Build the shared library (pjrt_qnn.dll on Windows, pjrt_qnn.so on Linux)
cmake --build . --config Release
```

If `QNN_SDK_ROOT` is not provided, CMake will build in **STUB mode** (with CPU reference execution for all supported StableHLO ops).

---

## 4. Installing the Python Package

Install `jax-qnn` in editable mode:

```bash
pip install -e .
```

---

## 5. Verifying Installation

Run Python:

```python
import jax
import jax_qnn

# Verify device enumeration
devices = jax.devices("qnn")
print(devices)
# Output: [QnnDevice(id=0)]

# Run a test calculation
@jax.jit
def f(x, y):
    return x + y

import jax.numpy as jnp
x = jnp.ones((4, 4))
y = jnp.ones((4, 4))

out = jax.jit(f, backend="qnn")(x, y)
print(out)
```

---

## 6. Running Tests

```bash
pytest tests/ -v
```
