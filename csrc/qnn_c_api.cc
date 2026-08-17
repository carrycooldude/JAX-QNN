// =============================================================================
// qnn_c_api.cc — Direct C ABI implementation for Python ctypes bridge
// =============================================================================

#include "qnn_c_api.h"
#include "qnn_types.h"
#include "qnn_runtime.h"
#include "stablehlo_to_qnn.h"

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>

namespace jax_qnn {

// Forward declaration from qnn_executable.cc
PJRT_Error* ExecuteComputation(
    const ParsedComputation& comp,
    const std::vector<QnnBufferState*>& inputs,
    std::vector<std::unique_ptr<QnnBufferState>>& outputs,
    QnnClientState* client);

struct DirectExecutable {
  ParsedComputation computation;
  std::string name;
};

}  // namespace jax_qnn

extern "C" {

JAX_QNN_EXPORT int JAX_QNN_Init(int backend_type, const char* sdk_root) {
  jax_qnn::QnnRuntimeConfig config;
  if (backend_type == 1) {
    config.backend_type = jax_qnn::QnnBackendType::kHTP;
  } else if (backend_type == 2) {
    config.backend_type = jax_qnn::QnnBackendType::kGPU;
  } else if (backend_type == 3) {
    config.backend_type = jax_qnn::QnnBackendType::kCPU;
  } else {
    config.backend_type = jax_qnn::QnnBackendType::kAuto;
  }

  if (sdk_root && std::strlen(sdk_root) > 0) {
    config.sdk_root = sdk_root;
  }

  bool ok = jax_qnn::QnnRuntime::Instance().Initialize(config);
  return ok ? 0 : -1;
}

JAX_QNN_EXPORT const char* JAX_QNN_GetBackendName() {
  static std::string name;
  name = jax_qnn::QnnRuntime::Instance().GetBackendName();
  return name.c_str();
}

JAX_QNN_EXPORT void* JAX_QNN_Compile(const char* mlir_text, size_t text_len) {
  if (!mlir_text || text_len == 0) {
    return nullptr;
  }

  std::string code(mlir_text, text_len);
  auto parsed = jax_qnn::ParseStableHLO(code);
  if (!parsed) {
    return nullptr;
  }

  auto* exec = new jax_qnn::DirectExecutable();
  exec->computation = std::move(*parsed);
  exec->name = exec->computation.name.empty() ? "qnn_direct_exec" : exec->computation.name;

  return reinterpret_cast<void*>(exec);
}

JAX_QNN_EXPORT void JAX_QNN_DestroyExecutable(void* exec_handle) {
  if (!exec_handle) return;
  auto* exec = reinterpret_cast<jax_qnn::DirectExecutable*>(exec_handle);
  delete exec;
}

JAX_QNN_EXPORT int JAX_QNN_GetNumInputs(void* exec_handle) {
  if (!exec_handle) return 0;
  auto* exec = reinterpret_cast<jax_qnn::DirectExecutable*>(exec_handle);
  return exec->computation.num_inputs;
}

JAX_QNN_EXPORT int JAX_QNN_GetNumOutputs(void* exec_handle) {
  if (!exec_handle) return 0;
  auto* exec = reinterpret_cast<jax_qnn::DirectExecutable*>(exec_handle);
  return exec->computation.num_outputs;
}

JAX_QNN_EXPORT int64_t JAX_QNN_GetOutputByteSize(void* exec_handle, int output_idx) {
  if (!exec_handle) return 0;
  auto* exec = reinterpret_cast<jax_qnn::DirectExecutable*>(exec_handle);
  if (output_idx < 0 || output_idx >= static_cast<int>(exec->computation.output_shapes.size())) {
    return 0;
  }
  
  int64_t num_elements = 1;
  for (auto d : exec->computation.output_shapes[output_idx]) {
    num_elements *= d;
  }
  
  size_t elem_size = 4; // float32 default
  if (output_idx < static_cast<int>(exec->computation.output_types.size())) {
    elem_size = jax_qnn::PjrtTypeByteSize(exec->computation.output_types[output_idx]);
  }
  return num_elements * elem_size;
}

JAX_QNN_EXPORT int JAX_QNN_GetOutputRank(void* exec_handle, int output_idx) {
  if (!exec_handle) return 0;
  auto* exec = reinterpret_cast<jax_qnn::DirectExecutable*>(exec_handle);
  if (output_idx < 0 || output_idx >= static_cast<int>(exec->computation.output_shapes.size())) {
    return 0;
  }
  return static_cast<int>(exec->computation.output_shapes[output_idx].size());
}

JAX_QNN_EXPORT int JAX_QNN_GetOutputShape(void* exec_handle, int output_idx, int64_t* dims_out) {
  if (!exec_handle || !dims_out) return -1;
  auto* exec = reinterpret_cast<jax_qnn::DirectExecutable*>(exec_handle);
  if (output_idx < 0 || output_idx >= static_cast<int>(exec->computation.output_shapes.size())) {
    return -1;
  }
  const auto& shape = exec->computation.output_shapes[output_idx];
  for (size_t i = 0; i < shape.size(); ++i) {
    dims_out[i] = shape[i];
  }
  return 0;
}

JAX_QNN_EXPORT int JAX_QNN_Execute(
    void* exec_handle,
    const void* const* input_ptrs,
    const int64_t* input_bytes,
    int num_inputs,
    void* const* output_ptrs,
    const int64_t* output_bytes,
    int num_outputs
) {
  if (!exec_handle) return -1;
  auto* exec = reinterpret_cast<jax_qnn::DirectExecutable*>(exec_handle);

  // Wrap inputs as QnnBufferState views
  std::vector<jax_qnn::QnnBufferState*> input_bufs;
  std::vector<std::unique_ptr<jax_qnn::QnnBufferState>> owned_input_bufs;

  for (int i = 0; i < num_inputs; ++i) {
    auto buf = std::make_unique<jax_qnn::QnnBufferState>();
    if (i < static_cast<int>(exec->computation.input_shapes.size())) {
      buf->dimensions = exec->computation.input_shapes[i];
      buf->element_type = exec->computation.input_types[i];
    }
    
    // Copy data from input_ptrs[i]
    size_t size = static_cast<size_t>(input_bytes[i]);
    buf->data.resize(size);
    if (input_ptrs[i] && size > 0) {
      std::memcpy(buf->data.data(), input_ptrs[i], size);
    }
    buf->is_ready.store(true);
    input_bufs.push_back(buf.get());
    owned_input_bufs.push_back(std::move(buf));
  }

  // Execute computation
  std::vector<std::unique_ptr<jax_qnn::QnnBufferState>> output_bufs;
  ::PJRT_Error* err = jax_qnn::ExecuteComputation(
      exec->computation, input_bufs, output_bufs, nullptr);

  if (err) {
    return -2;
  }

  // Copy outputs to caller-provided output_ptrs
  for (int i = 0; i < num_outputs && i < static_cast<int>(output_bufs.size()); ++i) {
    if (output_ptrs[i] && output_bytes[i] > 0) {
      size_t copy_size = std::min(static_cast<size_t>(output_bytes[i]), output_bufs[i]->data.size());
      std::memcpy(output_ptrs[i], output_bufs[i]->data.data(), copy_size);
    }
  }

  return 0;
}

}  // extern "C"
