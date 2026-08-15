// =============================================================================
// qnn_executable.cc — PJRT Executable and LoadedExecutable for QNN
// =============================================================================
//
// This file handles compilation output (PJRT_Executable) and execution
// (PJRT_LoadedExecutable_Execute). When JAX calls Execute, we:
//
// 1. Extract input data from PJRT Buffers
// 2. In real QNN mode: feed into QnnGraph_execute()
// 3. In stub mode: execute operations on CPU as a reference implementation
// 4. Wrap outputs as PJRT Buffers
//
// The stub mode is critical for development — it allows testing the full
// JAX → StableHLO → parse → execute pipeline without the QNN SDK.
//
// =============================================================================

#include "qnn_types.h"
#include "pjrt/pjrt_c_api.h"

#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace jax_qnn {

// =============================================================================
// Stub execution engine — runs ops on CPU for development/testing
// =============================================================================

namespace stub {

// Element-wise binary op on f32 buffers
static void BinaryOp(const float* a, const float* b, float* out,
                     size_t num_elements, const std::string& op_type) {
  for (size_t i = 0; i < num_elements; ++i) {
    if (op_type == "stablehlo.add") {
      out[i] = a[i] + b[i];
    } else if (op_type == "stablehlo.subtract") {
      out[i] = a[i] - b[i];
    } else if (op_type == "stablehlo.multiply") {
      out[i] = a[i] * b[i];
    } else if (op_type == "stablehlo.divide") {
      out[i] = (b[i] != 0.0f) ? a[i] / b[i] : 0.0f;
    } else if (op_type == "stablehlo.maximum") {
      out[i] = std::max(a[i], b[i]);
    } else if (op_type == "stablehlo.minimum") {
      out[i] = std::min(a[i], b[i]);
    }
  }
}

// Matrix multiply: C[M,N] = A[M,K] @ B[K,N]
static void MatMul(const float* a, const float* b, float* out,
                   int64_t M, int64_t K, int64_t N) {
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += a[m * K + k] * b[k * N + n];
      }
      out[m * N + n] = sum;
    }
  }
}

// ReLU: out = max(0, x)
static void Relu(const float* x, float* out, size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    out[i] = std::max(0.0f, x[i]);
  }
}

// Negate: out = -x
static void Negate(const float* x, float* out, size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    out[i] = -x[i];
  }
}

// Reduce sum along specified axes
static void ReduceSum(const float* input, float* output,
                      const std::vector<int64_t>& shape,
                      const std::vector<int>& reduction_dims) {
  // Simple case: reduce all dims to scalar
  size_t total = 1;
  for (auto d : shape) total *= d;
  float sum = 0.0f;
  for (size_t i = 0; i < total; ++i) {
    sum += input[i];
  }
  output[0] = sum;
}

}  // namespace stub

// =============================================================================
// Execute the parsed computation
// =============================================================================

static PJRT_Error* ExecuteComputation(
    const ParsedComputation& comp,
    const std::vector<QnnBufferState*>& inputs,
    std::vector<std::unique_ptr<QnnBufferState>>& outputs,
    QnnClientState* client) {
  
  // Value table: maps value indices to data pointers
  // First N entries are the inputs
  std::vector<std::vector<uint8_t>> value_storage;
  std::vector<float*> value_ptrs;
  std::vector<std::vector<int64_t>> value_shapes;
  std::vector<PJRT_Buffer_Type> value_types;
  
  // Initialize with input values
  for (size_t i = 0; i < inputs.size(); ++i) {
    value_storage.push_back(inputs[i]->data);
    value_ptrs.push_back(reinterpret_cast<float*>(value_storage.back().data()));
    value_shapes.push_back(inputs[i]->dimensions);
    value_types.push_back(inputs[i]->element_type);
  }
  
  // Execute ops sequentially
  for (const auto& op : comp.ops) {
    if (op.op_type == "stablehlo.return") {
      // Return op — marks the outputs. The output indices refer to
      // values in our value table.
      continue;
    }
    
    // Compute output shape and allocate
    std::vector<int64_t> out_shape;
    PJRT_Buffer_Type out_type = PJRT_Buffer_Type_F32;
    
    if (!op.output_shapes.empty()) {
      out_shape = op.output_shapes[0];
      out_type = op.output_types.empty() ? PJRT_Buffer_Type_F32 : op.output_types[0];
    } else if (!op.input_indices.empty() && 
               op.input_indices[0] < (int)value_shapes.size()) {
      out_shape = value_shapes[op.input_indices[0]];
      out_type = value_types[op.input_indices[0]];
    }
    
    size_t num_elements = 1;
    for (auto d : out_shape) num_elements *= d;
    size_t byte_size = num_elements * PjrtTypeByteSize(out_type);
    
    value_storage.emplace_back(byte_size, 0);
    float* out_ptr = reinterpret_cast<float*>(value_storage.back().data());
    value_ptrs.push_back(out_ptr);
    value_shapes.push_back(out_shape);
    value_types.push_back(out_type);
    
    // Dispatch by op type
    if (op.op_type == "stablehlo.add" || op.op_type == "stablehlo.subtract" ||
        op.op_type == "stablehlo.multiply" || op.op_type == "stablehlo.divide" ||
        op.op_type == "stablehlo.maximum" || op.op_type == "stablehlo.minimum") {
      if (op.input_indices.size() >= 2) {
        int idx_a = op.input_indices[0];
        int idx_b = op.input_indices[1];
        if (idx_a < (int)value_ptrs.size() && idx_b < (int)value_ptrs.size()) {
          stub::BinaryOp(value_ptrs[idx_a], value_ptrs[idx_b], 
                         out_ptr, num_elements, op.op_type);
        }
      }
    } else if (op.op_type == "stablehlo.dot_general" || 
               op.op_type == "stablehlo.dot") {
      if (op.input_indices.size() >= 2) {
        int idx_a = op.input_indices[0];
        int idx_b = op.input_indices[1];
        if (idx_a < (int)value_ptrs.size() && idx_b < (int)value_ptrs.size()) {
          auto& shape_a = value_shapes[idx_a];
          auto& shape_b = value_shapes[idx_b];
          int64_t M = shape_a.size() >= 2 ? shape_a[shape_a.size()-2] : 1;
          int64_t K = shape_a.size() >= 1 ? shape_a[shape_a.size()-1] : 1;
          int64_t N = shape_b.size() >= 1 ? shape_b[shape_b.size()-1] : 1;
          stub::MatMul(value_ptrs[idx_a], value_ptrs[idx_b], 
                       out_ptr, M, K, N);
        }
      }
    } else if (op.op_type == "stablehlo.negate") {
      if (!op.input_indices.empty()) {
        int idx = op.input_indices[0];
        if (idx < (int)value_ptrs.size()) {
          stub::Negate(value_ptrs[idx], out_ptr, num_elements);
        }
      }
    } else if (op.op_type == "stablehlo.reshape") {
      // Reshape is a no-op on the data, just changes shape metadata
      if (!op.input_indices.empty()) {
        int idx = op.input_indices[0];
        if (idx < (int)value_ptrs.size()) {
          std::memcpy(out_ptr, value_ptrs[idx], byte_size);
        }
      }
    } else if (op.op_type == "stablehlo.transpose") {
      // TODO: implement proper transpose
      if (!op.input_indices.empty()) {
        int idx = op.input_indices[0];
        if (idx < (int)value_ptrs.size()) {
          std::memcpy(out_ptr, value_ptrs[idx], byte_size);
        }
      }
    } else if (op.op_type == "stablehlo.broadcast_in_dim") {
      // Simple broadcast: replicate the input data
      if (!op.input_indices.empty()) {
        int idx = op.input_indices[0];
        if (idx < (int)value_ptrs.size()) {
          size_t in_elements = 1;
          for (auto d : value_shapes[idx]) in_elements *= d;
          // Simple case: scalar broadcast
          if (in_elements == 1) {
            float val = value_ptrs[idx][0];
            for (size_t i = 0; i < num_elements; ++i) out_ptr[i] = val;
          } else {
            // General broadcast — copy what fits
            std::memcpy(out_ptr, value_ptrs[idx], 
                       std::min(byte_size, in_elements * sizeof(float)));
          }
        }
      }
    } else if (op.op_type == "stablehlo.reduce") {
      // Simplified: reduce sum
      if (!op.input_indices.empty()) {
        int idx = op.input_indices[0];
        if (idx < (int)value_ptrs.size()) {
          stub::ReduceSum(value_ptrs[idx], out_ptr, 
                         value_shapes[idx], {});
        }
      }
    } else if (op.op_type == "stablehlo.constant") {
      // Constants are embedded in the computation
      // The data should already be in the output_shapes/types
      // For now, zero-initialize
    } else {
      std::cerr << "[JAX-QNN] WARNING: Unsupported op '" << op.op_type 
                << "', zero-filling output." << std::endl;
    }
  }
  
  // Collect outputs from the last N values in the value table
  // (corresponding to the computation's outputs)
  for (size_t i = 0; i < comp.num_outputs; ++i) {
    auto out_buf = std::make_unique<QnnBufferState>();
    out_buf->client = client;
    out_buf->device = &client->device;
    out_buf->memory = &client->memory;
    
    if (i < comp.output_shapes.size()) {
      out_buf->dimensions = comp.output_shapes[i];
      out_buf->element_type = comp.output_types[i];
    }
    
    // Get data from the last values produced
    size_t value_idx = value_storage.size() - comp.num_outputs + i;
    if (value_idx < value_storage.size()) {
      out_buf->data = std::move(value_storage[value_idx]);
    } else {
      out_buf->data.resize(out_buf->GetByteSize(), 0);
    }
    
    out_buf->is_ready.store(true);
    outputs.push_back(std::move(out_buf));
  }
  
  return NoError();
}

// =============================================================================
// PJRT Executable API
// =============================================================================

PJRT_Error* QnnExecutableName(PJRT_Executable_Name_Args* args) {
  auto* exec = reinterpret_cast<QnnExecutableState*>(args->executable);
  args->executable_name = exec->name.c_str();
  args->executable_name_size = exec->name.size();
  return NoError();
}

PJRT_Error* QnnExecutableNumOutputs(PJRT_Executable_NumOutputs_Args* args) {
  auto* exec = reinterpret_cast<QnnExecutableState*>(args->executable);
  args->num_outputs = exec->computation.num_outputs;
  return NoError();
}

PJRT_Error* QnnExecutableDestroy(PJRT_Executable_Destroy_Args* args) {
  auto* exec = reinterpret_cast<QnnExecutableState*>(args->executable);
  delete exec;
  return NoError();
}

PJRT_Error* QnnLoadedExecutableDestroy(
    PJRT_LoadedExecutable_Destroy_Args* args) {
  auto* loaded = reinterpret_cast<QnnLoadedExecutableState*>(args->executable);
  if (loaded->executable) {
    delete loaded->executable;
  }
  delete loaded;
  return NoError();
}

PJRT_Error* QnnLoadedExecutableGetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* args) {
  auto* loaded = reinterpret_cast<QnnLoadedExecutableState*>(
      args->loaded_executable);
  args->executable = reinterpret_cast<PJRT_Executable*>(loaded->executable);
  return NoError();
}

PJRT_Error* QnnLoadedExecutableAddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args) {
  auto* loaded = reinterpret_cast<QnnLoadedExecutableState*>(args->executable);
  if (!loaded->addressable_devices.empty()) {
    args->addressable_devices = reinterpret_cast<PJRT_Device* const*>(
        loaded->addressable_devices.data());
    args->num_addressable_devices = loaded->addressable_devices.size();
  } else {
    args->addressable_devices = nullptr;
    args->num_addressable_devices = 0;
  }
  return NoError();
}

PJRT_Error* QnnLoadedExecutableDelete(
    PJRT_LoadedExecutable_Delete_Args* args) {
  auto* loaded = reinterpret_cast<QnnLoadedExecutableState*>(args->executable);
  loaded->is_deleted = true;
  return NoError();
}

PJRT_Error* QnnLoadedExecutableIsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args* args) {
  auto* loaded = reinterpret_cast<QnnLoadedExecutableState*>(args->executable);
  args->is_deleted = loaded->is_deleted;
  return NoError();
}

// =============================================================================
// Execute — the core execution function
// =============================================================================

PJRT_Error* QnnLoadedExecutableExecute(
    PJRT_LoadedExecutable_Execute_Args* args) {
  auto* loaded = reinterpret_cast<QnnLoadedExecutableState*>(args->executable);
  auto* exec = loaded->executable;
  
  if (!exec || !exec->is_compiled) {
    return MakeError(PJRT_Error_Code_FAILED_PRECONDITION,
        "QNN execute: executable not compiled");
  }
  
  std::cerr << "[JAX-QNN] Executing '" << exec->name << "' with "
            << args->num_args << " args on " << args->num_devices 
            << " device(s)." << std::endl;
  
  // For each device (we only have 1)
  for (size_t dev = 0; dev < args->num_devices; ++dev) {
    // Gather input buffers
    std::vector<QnnBufferState*> input_buffers;
    for (size_t i = 0; i < args->num_args; ++i) {
      auto* buf = reinterpret_cast<QnnBufferState*>(
          args->argument_lists[dev][i]);
      input_buffers.push_back(buf);
    }
    
    // Execute
    std::vector<std::unique_ptr<QnnBufferState>> output_buffers;
    PJRT_Error* err = ExecuteComputation(
        exec->computation, input_buffers, output_buffers, loaded->client);
    if (err) return err;
    
    // Copy outputs to the output_lists array
    for (size_t i = 0; i < output_buffers.size() && 
         i < exec->computation.num_outputs; ++i) {
      args->output_lists[dev][i] = reinterpret_cast<PJRT_Buffer*>(
          output_buffers[i].release());
    }
    
    // Create completion event
    if (args->device_complete_events) {
      auto* event = new QnnEventState();
      event->is_ready.store(true);
      args->device_complete_events[dev] = reinterpret_cast<PJRT_Event*>(event);
    }
  }
  
  return NoError();
}

}  // namespace jax_qnn
