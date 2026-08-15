// =============================================================================
// qnn_runtime.cc — Qualcomm QNN SDK Runtime Interface Implementation
// =============================================================================

#include "qnn_runtime.h"

#include <iostream>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#define LOAD_LIB(path) LoadLibraryA(path)
#define GET_SYM(handle, name) GetProcAddress((HMODULE)handle, name)
#define CLOSE_LIB(handle) FreeLibrary((HMODULE)handle)
#else
#include <dlfcn.h>
#define LOAD_LIB(path) dlopen(path, RTLD_NOW | RTLD_LOCAL)
#define GET_SYM(handle, name) dlsym(handle, name)
#define CLOSE_LIB(handle) dlclose(handle)
#endif

namespace jax_qnn {

QnnRuntime& QnnRuntime::Instance() {
  static QnnRuntime instance;
  return instance;
}

QnnRuntime::~QnnRuntime() {
  Shutdown();
}

bool QnnRuntime::Initialize(const QnnRuntimeConfig& config) {
  if (initialized_) {
    return true;
  }

  std::string backend_lib = config.custom_backend_path;

  if (backend_lib.empty()) {
    std::string root = config.sdk_root;
    if (root.empty()) {
      const char* env_root = std::getenv("QNN_SDK_ROOT");
      if (env_root) root = env_root;
    }

    if (!root.empty()) {
#ifdef _WIN32
      std::string htp_lib = root + "\\lib\\aarch64-windows-msvc\\QnnHtp.dll";
      std::string cpu_lib = root + "\\lib\\aarch64-windows-msvc\\QnnCpu.dll";
#else
      std::string htp_lib = root + "/lib/aarch64-linux-clang/libQnnHtp.so";
      std::string cpu_lib = root + "/lib/aarch64-linux-clang/libQnnCpu.so";
#endif
      if (config.backend_type == QnnBackendType::kHTP) {
        backend_lib = htp_lib;
      } else if (config.backend_type == QnnBackendType::kCPU) {
        backend_lib = cpu_lib;
      } else {
        // Try HTP first, fallback to CPU
        backend_lib = htp_lib;
      }
    }
  }

  if (!backend_lib.empty() && LoadBackendLibrary(backend_lib)) {
    is_stub_mode_ = false;
    initialized_ = true;
    std::cerr << "[JAX-QNN] Successfully loaded QNN backend: " << backend_lib << std::endl;
    return true;
  }

  // Fallback to stub mode
  is_stub_mode_ = true;
  backend_name_ = "qnn_cpu_stub";
  backend_version_ = "0.1.0-stub";
  initialized_ = true;
  std::cerr << "[JAX-QNN] Running in stub mode (reference CPU execution for StableHLO)." << std::endl;
  return true;
}

bool QnnRuntime::LoadBackendLibrary(const std::string& lib_path) {
  void* handle = LOAD_LIB(lib_path.c_str());
  if (!handle) {
    std::cerr << "[JAX-QNN] Could not open library: " << lib_path << std::endl;
    return false;
  }

  lib_handle_ = handle;
  backend_name_ = lib_path;
  backend_version_ = "Qualcomm QNN SDK";
  return true;
}

void QnnRuntime::Shutdown() {
  if (lib_handle_) {
    CLOSE_LIB(lib_handle_);
    lib_handle_ = nullptr;
  }
  initialized_ = false;
  is_stub_mode_ = true;
}

}  // namespace jax_qnn
