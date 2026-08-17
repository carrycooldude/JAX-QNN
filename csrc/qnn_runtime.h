// =============================================================================
// qnn_runtime.h — Qualcomm QNN SDK Native Runtime Interface
// =============================================================================

#ifndef JAX_QNN_QNN_RUNTIME_H_
#define JAX_QNN_QNN_RUNTIME_H_

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>

#ifndef QNN_STUB_MODE
#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "QnnOpDef.h"
#endif

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

struct NativeQnnTensor {
  std::string name;
  std::vector<uint32_t> shape;
  int data_type = 0; // 0 = Float32, etc.
  std::vector<uint8_t> data;
  void* client_buf = nullptr;
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

  // Hardware Graph Management
  void* CreateHardwareGraph(const std::string& graph_name);
  bool FinalizeHardwareGraph(void* graph_handle);
  bool ExecuteHardwareGraph(
      void* graph_handle,
      const std::vector<void*>& input_ptrs,
      const std::vector<size_t>& input_bytes,
      const std::vector<void*>& output_ptrs,
      const std::vector<size_t>& output_bytes);
  void DestroyHardwareGraph(void* graph_handle);

 private:
  QnnRuntime() = default;
  ~QnnRuntime();

  bool LoadBackendLibrary(const std::string& lib_path);

  bool initialized_ = false;
  bool is_stub_mode_ = true;
  std::string backend_name_ = "stub";
  std::string backend_version_ = "0.1.0";
  void* lib_handle_ = nullptr;

#ifndef QNN_STUB_MODE
  Qnn_BackendHandle_t backend_handle_ = nullptr;
  Qnn_DeviceHandle_t device_handle_ = nullptr;
  Qnn_ContextHandle_t context_handle_ = nullptr;
  QnnInterface_t qnn_interface_;
  bool has_interface_ = false;
#endif
};

}  // namespace jax_qnn

#endif  // JAX_QNN_QNN_RUNTIME_H_
