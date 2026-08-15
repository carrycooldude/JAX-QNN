// =============================================================================
// qnn_device.cc — PJRT Device and DeviceDescription implementation for QNN
// =============================================================================

#include "qnn_types.h"
#include "pjrt/pjrt_c_api.h"

namespace jax_qnn {

// =============================================================================
// DeviceDescription
// =============================================================================

PJRT_Error* QnnDeviceDescriptionId(PJRT_DeviceDescription_Id_Args* args) {
  auto* desc = reinterpret_cast<QnnDeviceState*>(args->device_description);
  args->id = desc->id;
  return NoError();
}

PJRT_Error* QnnDeviceDescriptionProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args* args) {
  args->process_index = 0;  // Single process
  return NoError();
}

PJRT_Error* QnnDeviceDescriptionAttributes(
    PJRT_DeviceDescription_Attributes_Args* args) {
  args->num_attributes = 0;
  args->attributes = nullptr;
  return NoError();
}

PJRT_Error* QnnDeviceDescriptionKind(
    PJRT_DeviceDescription_Kind_Args* args) {
  auto* desc = reinterpret_cast<QnnDeviceState*>(args->device_description);
  args->device_kind = desc->device_kind.c_str();
  args->device_kind_size = desc->device_kind.size();
  return NoError();
}

PJRT_Error* QnnDeviceDescriptionDebugString(
    PJRT_DeviceDescription_DebugString_Args* args) {
  auto* desc = reinterpret_cast<QnnDeviceState*>(args->device_description);
  args->debug_string = desc->debug_string.c_str();
  args->debug_string_size = desc->debug_string.size();
  return NoError();
}

PJRT_Error* QnnDeviceDescriptionToString(
    PJRT_DeviceDescription_ToString_Args* args) {
  auto* desc = reinterpret_cast<QnnDeviceState*>(args->device_description);
  args->to_string = desc->to_string.c_str();
  args->to_string_size = desc->to_string.size();
  return NoError();
}

// =============================================================================
// Device
// =============================================================================

PJRT_Error* QnnDeviceGetDescription(PJRT_Device_GetDescription_Args* args) {
  // The device and its description are the same object in our implementation
  auto* device = reinterpret_cast<QnnDeviceState*>(args->device);
  args->device_description = reinterpret_cast<PJRT_DeviceDescription*>(device);
  return NoError();
}

PJRT_Error* QnnDeviceIsAddressable(PJRT_Device_IsAddressable_Args* args) {
  args->is_addressable = true;
  return NoError();
}

PJRT_Error* QnnDeviceLocalHardwareId(
    PJRT_Device_LocalHardwareId_Args* args) {
  auto* device = reinterpret_cast<QnnDeviceState*>(args->device);
  args->local_hardware_id = device->local_hardware_id;
  return NoError();
}

PJRT_Error* QnnDeviceAddressableMemories(
    PJRT_Device_AddressableMemories_Args* args) {
  auto* device = reinterpret_cast<QnnDeviceState*>(args->device);
  args->memories = reinterpret_cast<PJRT_Memory* const*>(
      &device->client->memory_ptr);
  args->num_memories = 1;
  return NoError();
}

PJRT_Error* QnnDeviceDefaultMemory(PJRT_Device_DefaultMemory_Args* args) {
  auto* device = reinterpret_cast<QnnDeviceState*>(args->device);
  args->memory = reinterpret_cast<PJRT_Memory*>(&device->client->memory);
  return NoError();
}

// =============================================================================
// Memory
// =============================================================================

PJRT_Error* QnnMemoryId(PJRT_Memory_Id_Args* args) {
  auto* mem = reinterpret_cast<QnnMemoryState*>(args->memory);
  args->id = mem->id;
  return NoError();
}

PJRT_Error* QnnMemoryKind(PJRT_Memory_Kind_Args* args) {
  auto* mem = reinterpret_cast<QnnMemoryState*>(args->memory);
  args->kind = mem->kind.c_str();
  args->kind_size = mem->kind.size();
  return NoError();
}

PJRT_Error* QnnMemoryKindId(PJRT_Memory_Kind_Id_Args* args) {
  auto* mem = reinterpret_cast<QnnMemoryState*>(args->memory);
  args->kind_id = mem->kind_id;
  return NoError();
}

PJRT_Error* QnnMemoryDebugString(PJRT_Memory_DebugString_Args* args) {
  auto* mem = reinterpret_cast<QnnMemoryState*>(args->memory);
  args->debug_string = mem->debug_string.c_str();
  args->debug_string_size = mem->debug_string.size();
  return NoError();
}

PJRT_Error* QnnMemoryToString(PJRT_Memory_ToString_Args* args) {
  auto* mem = reinterpret_cast<QnnMemoryState*>(args->memory);
  args->to_string = mem->to_string.c_str();
  args->to_string_size = mem->to_string.size();
  return NoError();
}

PJRT_Error* QnnMemoryAddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args* args) {
  auto* mem = reinterpret_cast<QnnMemoryState*>(args->memory);
  auto* client = mem->device->client;
  args->devices = reinterpret_cast<PJRT_Device* const*>(&client->device_ptr);
  args->num_devices = 1;
  return NoError();
}

}  // namespace jax_qnn
