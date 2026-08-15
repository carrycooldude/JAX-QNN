// =============================================================================
// qnn_client.cc — PJRT Client implementation for QNN
// =============================================================================
//
// This file implements the PJRT_Client_* functions that JAX calls to create
// the QNN backend client, enumerate devices, compile programs, and create
// buffers from host data.
//
// The client is the top-level object in the PJRT hierarchy. When JAX calls
// register_plugin("qnn", library_path="pjrt_qnn.dll"), the following happens:
//
//   1. JAX loads the DLL and calls GetPjRtApi()
//   2. JAX calls PJRT_Client_Create to create a new client
//   3. JAX calls PJRT_Client_Devices to enumerate available QNN devices
//   4. When jax.jit(f, backend="qnn") is called, JAX calls PJRT_Client_Compile
//   5. When execution happens, JAX calls PJRT_LoadedExecutable_Execute
//
// =============================================================================

#include "qnn_client.h"
#include "qnn_types.h"
#include "stablehlo_to_qnn.h"

#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>

namespace jax_qnn {

// =============================================================================
// Client Create / Destroy
// =============================================================================

PJRT_Error* QnnClientCreate(PJRT_Client_Create_Args* args) {
  // Allocate and initialize the client state
  auto* state = new QnnClientState();
  
  // Parse create options if provided
  for (size_t i = 0; i < args->num_options; ++i) {
    const auto& opt = args->create_options[i];
    std::string name(opt.name, opt.name_size);
    
    if (name == "qnn_backend" && opt.type == PJRT_NamedValue_kString) {
      state->qnn_backend_lib = std::string(opt.string_value, opt.value_size);
    }
  }
  
  // Try to determine QNN backend library path
  if (state->qnn_backend_lib.empty()) {
    // Check environment variable
    const char* qnn_sdk_root = std::getenv("QNN_SDK_ROOT");
    if (qnn_sdk_root) {
#ifdef _WIN32
      state->qnn_backend_lib = std::string(qnn_sdk_root) + 
          "\\lib\\aarch64-windows-msvc\\QnnCpu.dll";
#else
      state->qnn_backend_lib = std::string(qnn_sdk_root) + 
          "/lib/aarch64-linux-clang/libQnnCpu.so";
#endif
    }
  }
  
  // Configure the device
  state->device.debug_string = "QnnDevice(id=0)";
  state->device.to_string = "QnnDevice(id=0, kind=qnn)";
  
  if (!state->qnn_backend_lib.empty()) {
    state->platform_version = "0.1.0 (QNN backend: " + 
        state->qnn_backend_lib + ")";
  }
  
  // Initialize QNN runtime if SDK is available
#ifndef QNN_STUB_MODE
  // TODO: Initialize QNN backend via qnn_runtime.cc
  // For now, we set qnn_initialized based on stub mode
#endif
  
  std::cerr << "[JAX-QNN] Client created. Platform: " << state->platform_name
            << ", Version: " << state->platform_version << std::endl;
  
  // Return the client to JAX via the opaque pointer
  args->client = reinterpret_cast<PJRT_Client*>(state);
  return NoError();
}

PJRT_Error* QnnClientDestroy(PJRT_Client_Destroy_Args* args) {
  auto* state = reinterpret_cast<QnnClientState*>(args->client);
  if (state) {
    std::cerr << "[JAX-QNN] Client destroyed." << std::endl;
    delete state;
  }
  return NoError();
}

// =============================================================================
// Platform Info
// =============================================================================

PJRT_Error* QnnClientPlatformName(PJRT_Client_PlatformName_Args* args) {
  auto* state = reinterpret_cast<QnnClientState*>(args->client);
  args->platform_name = state->platform_name.c_str();
  args->platform_name_size = state->platform_name.size();
  return NoError();
}

PJRT_Error* QnnClientProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
  auto* state = reinterpret_cast<QnnClientState*>(args->client);
  args->process_index = state->process_index;
  return NoError();
}

PJRT_Error* QnnClientPlatformVersion(PJRT_Client_PlatformVersion_Args* args) {
  auto* state = reinterpret_cast<QnnClientState*>(args->client);
  args->platform_version = state->platform_version.c_str();
  args->platform_version_size = state->platform_version.size();
  return NoError();
}

// =============================================================================
// Device Enumeration
// =============================================================================

PJRT_Error* QnnClientDevices(PJRT_Client_Devices_Args* args) {
  auto* state = reinterpret_cast<QnnClientState*>(args->client);
  // We expose exactly one QNN device
  args->devices = reinterpret_cast<PJRT_Device* const*>(&state->device_ptr);
  args->num_devices = 1;
  return NoError();
}

PJRT_Error* QnnClientAddressableDevices(
    PJRT_Client_AddressableDevices_Args* args) {
  auto* state = reinterpret_cast<QnnClientState*>(args->client);
  args->addressable_devices = 
      reinterpret_cast<PJRT_Device* const*>(&state->device_ptr);
  args->num_addressable_devices = 1;
  return NoError();
}

PJRT_Error* QnnClientLookupDevice(PJRT_Client_LookupDevice_Args* args) {
  auto* state = reinterpret_cast<QnnClientState*>(args->client);
  if (args->id == 0) {
    args->device = reinterpret_cast<PJRT_Device*>(&state->device);
    return NoError();
  }
  return MakeError(PJRT_Error_Code_NOT_FOUND, 
      "QNN device id " + std::to_string(args->id) + " not found. "
      "Only device id 0 is available.");
}

PJRT_Error* QnnClientLookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* args) {
  auto* state = reinterpret_cast<QnnClientState*>(args->client);
  if (args->local_hardware_id == 0) {
    args->addressable_device = reinterpret_cast<PJRT_Device*>(&state->device);
    return NoError();
  }
  return MakeError(PJRT_Error_Code_NOT_FOUND,
      "QNN addressable device with local_hardware_id " + 
      std::to_string(args->local_hardware_id) + " not found.");
}

// =============================================================================
// Compile — receives StableHLO, produces QNN Executable
// =============================================================================

PJRT_Error* QnnClientCompile(PJRT_Client_Compile_Args* args) {
  auto* client = reinterpret_cast<QnnClientState*>(args->client);
  
  if (!args->program || !args->program->code || args->program->code_size == 0) {
    return MakeError(PJRT_Error_Code_INVALID_ARGUMENT,
        "QNN compile: no program provided");
  }
  
  // Extract the program format and code
  std::string format(args->program->format, args->program->format_size);
  std::string code(args->program->code, args->program->code_size);
  
  std::cerr << "[JAX-QNN] Compiling program (" << code.size() << " bytes, "
            << "format: " << format << ")" << std::endl;
  
  // Parse the StableHLO/MLIR module
  auto computation = ParseStableHLO(code);
  if (!computation) {
    return MakeError(PJRT_Error_Code_INTERNAL,
        "QNN compile: failed to parse StableHLO program");
  }
  
  // Create executable state
  auto* exec_state = new QnnExecutableState();
  exec_state->client = client;
  exec_state->name = computation->name.empty() ? "qnn_executable" : computation->name;
  exec_state->computation = std::move(*computation);
  
  // Build QNN graph from parsed computation
  // In stub mode, we skip actual QNN graph construction
#ifndef QNN_STUB_MODE
  // TODO: Call QnnGraph_create(), QnnGraph_addNode(), QnnGraph_finalize()
  // via qnn_runtime.cc
#endif
  
  exec_state->is_compiled = true;
  
  // Create loaded executable state
  auto* loaded_state = new QnnLoadedExecutableState();
  loaded_state->client = client;
  loaded_state->executable = exec_state;
  loaded_state->addressable_devices.push_back(&client->device);
  
  args->executable = reinterpret_cast<PJRT_LoadedExecutable*>(loaded_state);
  
  std::cerr << "[JAX-QNN] Compilation successful. "
            << exec_state->computation.num_inputs << " inputs, "
            << exec_state->computation.num_outputs << " outputs, "
            << exec_state->computation.ops.size() << " ops." << std::endl;
  
  return NoError();
}

// =============================================================================
// BufferFromHostBuffer — creates a QNN buffer from host memory
// =============================================================================

PJRT_Error* QnnClientBufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args) {
  auto* client = reinterpret_cast<QnnClientState*>(args->client);
  
  // Create buffer state
  auto* buffer = new QnnBufferState();
  buffer->client = client;
  buffer->device = &client->device;
  buffer->memory = &client->memory;
  buffer->element_type = args->type;
  
  // Copy dimensions
  buffer->dimensions.assign(args->dims, args->dims + args->num_dims);
  
  // Calculate byte size and copy data
  size_t byte_size = buffer->GetByteSize();
  buffer->data.resize(byte_size);
  
  if (args->data && byte_size > 0) {
    // Handle byte strides if provided
    if (args->num_byte_strides > 0 && args->byte_strides) {
      // Strided copy — for now, do a simple contiguous copy
      // TODO: Handle non-contiguous strides properly
      std::memcpy(buffer->data.data(), args->data, byte_size);
    } else {
      std::memcpy(buffer->data.data(), args->data, byte_size);
    }
  }
  
  buffer->is_ready.store(true);
  
  // Return the buffer
  args->buffer = reinterpret_cast<PJRT_Buffer*>(buffer);
  
  // Create a "done with host buffer" event (immediately ready)
  if (args->done_with_host_buffer) {
    // Ignored for now — we always copy immediately
  }
  auto* event = new QnnEventState();
  event->is_ready.store(true);
  args->done_with_host_buffer = reinterpret_cast<PJRT_Event*>(event);
  
  return NoError();
}

}  // namespace jax_qnn
