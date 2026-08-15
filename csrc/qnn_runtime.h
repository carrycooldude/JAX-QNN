// =============================================================================
// qnn_runtime.h — Qualcomm QNN SDK Runtime Interface
// =============================================================================

#ifndef JAX_QNN_QNN_RUNTIME_H_
#define JAX_QNN_QNN_RUNTIME_H_

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace jax_qnn {

enum class QnnBackendType {
  kHTP,   // Hexagon Tensor Processor (NPU)
  kCPU,   // Reference CPU implementation
  kGPU,   // Adreno GPU
  kAuto,  // Automatically select best available
};

struct QnnRuntimeConfig {
  std::string sdk_root;
  QnnBackendType backend_type = QnnBackendType::kAuto;
  std::string custom_backend_path;
  int perf_profile = 0;  // 0: default, 1: high_performance, 2: power_saver
};

class QnnRuntime {
 public:
  static QnnRuntime& Instance();

  bool Initialize(const QnnRuntimeConfig& config);
  void Shutdown();

  bool IsInitialized() const { return initialized_; }
  bool IsStubMode() const { return is_stub_mode_; }

  std::string GetBackendName() const { return backend_name_; }
  std::string GetBackendVersion() const { return backend_version_; }

 private:
  QnnRuntime() = default;
  ~QnnRuntime();

  bool LoadBackendLibrary(const std::string& lib_path);

  bool initialized_ = false;
  bool is_stub_mode_ = true;
  std::string backend_name_ = "stub";
  std::string backend_version_ = "0.1.0";
  void* lib_handle_ = nullptr;
};

}  // namespace jax_qnn

#endif  // JAX_QNN_QNN_RUNTIME_H_
