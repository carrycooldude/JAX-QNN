// =============================================================================
// qnn_types.h — Internal types shared across the QNN PJRT plugin
// =============================================================================
//
// These structs are the "real" implementations behind the opaque PJRT C API
// pointers. When JAX passes us a PJRT_Client*, it's really a pointer to our
// QnnClientState. Same for Device, Buffer, Executable, etc.
//
// This file is the backbone of the plugin's internal state management.
// =============================================================================

#ifndef JAX_QNN_QNN_TYPES_H_
#define JAX_QNN_QNN_TYPES_H_

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <atomic>
#include <unordered_map>

#include "pjrt/pjrt_c_api.h"

namespace jax_qnn {

// =============================================================================
// Forward declarations
// =============================================================================

struct QnnClientState;
struct QnnDeviceState;
struct QnnMemoryState;
struct QnnBufferState;
struct QnnLoadedExecutableState;
struct QnnExecutableState;
struct QnnEventState;

// =============================================================================
// Error helpers
// =============================================================================

struct QnnError {
  PJRT_Error_Code code;
  std::string message;
};

// Creates a PJRT_Error from an error code and message.
// Returns nullptr if no error (success).
PJRT_Error* MakeError(PJRT_Error_Code code, const std::string& message);

// Returns nullptr (success).
inline PJRT_Error* NoError() { return nullptr; }

// =============================================================================
// Data type utilities
// =============================================================================

inline size_t PjrtTypeByteSize(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type_PRED: return 1;
    case PJRT_Buffer_Type_S8:   return 1;
    case PJRT_Buffer_Type_S16:  return 2;
    case PJRT_Buffer_Type_S32:  return 4;
    case PJRT_Buffer_Type_S64:  return 8;
    case PJRT_Buffer_Type_U8:   return 1;
    case PJRT_Buffer_Type_U16:  return 2;
    case PJRT_Buffer_Type_U32:  return 4;
    case PJRT_Buffer_Type_U64:  return 8;
    case PJRT_Buffer_Type_F16:  return 2;
    case PJRT_Buffer_Type_F32:  return 4;
    case PJRT_Buffer_Type_F64:  return 8;
    case PJRT_Buffer_Type_BF16: return 2;
    default: return 0;
  }
}

inline const char* PjrtTypeName(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type_F32:  return "f32";
    case PJRT_Buffer_Type_F64:  return "f64";
    case PJRT_Buffer_Type_F16:  return "f16";
    case PJRT_Buffer_Type_BF16: return "bf16";
    case PJRT_Buffer_Type_S8:   return "s8";
    case PJRT_Buffer_Type_S16:  return "s16";
    case PJRT_Buffer_Type_S32:  return "s32";
    case PJRT_Buffer_Type_S64:  return "s64";
    case PJRT_Buffer_Type_U8:   return "u8";
    case PJRT_Buffer_Type_U16:  return "u16";
    case PJRT_Buffer_Type_U32:  return "u32";
    case PJRT_Buffer_Type_U64:  return "u64";
    case PJRT_Buffer_Type_PRED: return "pred";
    default: return "unknown";
  }
}

// =============================================================================
// QNN Device State
// =============================================================================

struct QnnDeviceState {
  QnnClientState* client = nullptr;
  int id = 0;
  int local_hardware_id = 0;
  std::string device_kind = "qnn";
  std::string debug_string = "QnnDevice(id=0)";
  std::string to_string = "QnnDevice(id=0)";
  
  // Description (embedded, not a separate allocation)
  // PJRT_DeviceDescription points to this same struct cast as needed.
};

// =============================================================================
// QNN Memory State
// =============================================================================

struct QnnMemoryState {
  QnnDeviceState* device = nullptr;
  int id = 0;
  std::string kind = "qnn_shared";
  std::string debug_string = "QnnMemory(id=0, kind=qnn_shared)";
  std::string to_string = "QnnMemory(id=0, kind=qnn_shared)";
  int kind_id = 0;
};

// =============================================================================
// QNN Buffer State
// =============================================================================

struct QnnBufferState {
  QnnClientState* client = nullptr;
  QnnDeviceState* device = nullptr;
  QnnMemoryState* memory = nullptr;
  
  // Data storage
  std::vector<uint8_t> data;  // Host-side data copy
  PJRT_Buffer_Type element_type = PJRT_Buffer_Type_F32;
  std::vector<int64_t> dimensions;
  
  // State
  bool is_deleted = false;
  std::atomic<bool> is_ready{true};  // For sync; true = data available
  
  size_t GetByteSize() const {
    size_t elem_size = PjrtTypeByteSize(element_type);
    size_t num_elements = 1;
    for (int64_t d : dimensions) num_elements *= d;
    return num_elements * elem_size;
  }
};

// =============================================================================
// QNN Executable State
// =============================================================================

// Represents a single StableHLO operation that we've parsed and will
// translate to a QNN graph node.
struct ParsedOp {
  std::string op_type;         // e.g., "stablehlo.add", "stablehlo.dot_general"
  std::vector<int> input_indices;   // Indices into the computation's value list
  std::vector<int> output_indices;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<PJRT_Buffer_Type> output_types;
  
  // Additional attributes for specific ops
  std::unordered_map<std::string, std::string> attributes;
};

// Parsed StableHLO computation.
struct ParsedComputation {
  std::string name;
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  
  // Input/output metadata
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<PJRT_Buffer_Type> input_types;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<PJRT_Buffer_Type> output_types;
  
  // The sequence of operations
  std::vector<ParsedOp> ops;
  
  // The raw StableHLO/MLIR text (for debugging)
  std::string raw_mlir;
};

struct QnnExecutableState {
  QnnClientState* client = nullptr;
  std::string name = "qnn_executable";
  bool is_deleted = false;
  
  // The parsed computation
  ParsedComputation computation;
  
  // QNN handles (opaque, managed by qnn_runtime)
  void* qnn_graph_handle = nullptr;
  void* qnn_context_handle = nullptr;
  bool is_compiled = false;
};

struct QnnLoadedExecutableState {
  QnnClientState* client = nullptr;
  QnnExecutableState* executable = nullptr;
  bool is_deleted = false;
  
  // Devices this executable is loaded on
  std::vector<QnnDeviceState*> addressable_devices;
};

// =============================================================================
// QNN Event State
// =============================================================================

struct QnnEventState {
  std::atomic<bool> is_ready{true};
  PJRT_Error* error = nullptr;
  
  // Callback support
  std::mutex callback_mutex;
  std::vector<std::pair<PJRT_Event_OnReadyCallback, void*>> callbacks;
  
  void MarkReady(PJRT_Error* err = nullptr) {
    error = err;
    is_ready.store(true);
    std::lock_guard<std::mutex> lock(callback_mutex);
    for (auto& [cb, arg] : callbacks) {
      cb(err, arg);
    }
    callbacks.clear();
  }
};

// =============================================================================
// QNN Client State — the top-level state for the plugin
// =============================================================================

struct QnnClientState {
  // Platform identity
  std::string platform_name = "qnn";
  std::string platform_version = "0.1.0 (JAX-QNN PJRT Plugin)";
  int process_index = 0;
  
  // Devices (we expose exactly one QNN device)
  QnnDeviceState device;
  QnnDeviceState* device_ptr = nullptr;  // For PJRT_Client_Devices array
  
  // Memory spaces
  QnnMemoryState memory;
  QnnMemoryState* memory_ptr = nullptr;
  
  // QNN runtime state
  void* qnn_backend_handle = nullptr;
  std::string qnn_backend_lib;  // Path to QnnHtp.dll or QnnCpu.dll
  bool qnn_initialized = false;
  
  // Plugin attributes
  std::vector<PJRT_NamedValue> attributes;
  
  QnnClientState() {
    device.client = this;
    device.id = 0;
    device.local_hardware_id = 0;
    device_ptr = &device;
    
    memory.device = &device;
    memory.id = 0;
    memory_ptr = &memory;
  }
};

}  // namespace jax_qnn

#endif  // JAX_QNN_QNN_TYPES_H_
