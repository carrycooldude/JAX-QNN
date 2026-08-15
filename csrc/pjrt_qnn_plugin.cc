// =============================================================================
// pjrt_qnn_plugin.cc — Main dispatch table implementation for QNN PJRT plugin
// =============================================================================

#include "pjrt_qnn_plugin.h"
#include "qnn_types.h"
#include "qnn_client.h"

#include <iostream>
#include <string>
#include <cstring>

namespace jax_qnn {

// =============================================================================
// Error implementation
// =============================================================================

PJRT_Error* MakeError(PJRT_Error_Code code, const std::string& message) {
  auto* err = new QnnError{code, message};
  return reinterpret_cast<PJRT_Error*>(err);
}

static PJRT_Error* QnnErrorDestroy(PJRT_Error_Destroy_Args* args) {
  auto* err = reinterpret_cast<QnnError*>(args->error);
  delete err;
  return nullptr;
}

static PJRT_Error* QnnErrorMessage(PJRT_Error_Message_Args* args) {
  auto* err = reinterpret_cast<QnnError*>(args->error);
  args->message = err->message.c_str();
  args->message_size = err->message.size();
  return nullptr;
}

static PJRT_Error* QnnErrorGetCode(PJRT_Error_GetCode_Args* args) {
  auto* err = reinterpret_cast<QnnError*>(args->error);
  args->code = err->code;
  return nullptr;
}

// =============================================================================
// Plugin init / attributes
// =============================================================================

static PJRT_Error* QnnPluginInitialize(PJRT_Plugin_Initialize_Args* /*args*/) {
  std::cerr << "[JAX-QNN] PJRT Plugin Initialized." << std::endl;
  return nullptr;
}

static PJRT_Error* QnnPluginAttributes(PJRT_Plugin_Attributes_Args* args) {
  args->num_attributes = 0;
  args->attributes = nullptr;
  return nullptr;
}

// =============================================================================
// Forward declare declarations from other cc files
// =============================================================================

// Device & Memory (from qnn_device.cc)
PJRT_Error* QnnDeviceDescriptionId(PJRT_DeviceDescription_Id_Args* args);
PJRT_Error* QnnDeviceDescriptionProcessIndex(PJRT_DeviceDescription_ProcessIndex_Args* args);
PJRT_Error* QnnDeviceDescriptionAttributes(PJRT_DeviceDescription_Attributes_Args* args);
PJRT_Error* QnnDeviceDescriptionKind(PJRT_DeviceDescription_Kind_Args* args);
PJRT_Error* QnnDeviceDescriptionDebugString(PJRT_DeviceDescription_DebugString_Args* args);
PJRT_Error* QnnDeviceDescriptionToString(PJRT_DeviceDescription_ToString_Args* args);
PJRT_Error* QnnDeviceGetDescription(PJRT_Device_GetDescription_Args* args);
PJRT_Error* QnnDeviceIsAddressable(PJRT_Device_IsAddressable_Args* args);
PJRT_Error* QnnDeviceLocalHardwareId(PJRT_Device_LocalHardwareId_Args* args);
PJRT_Error* QnnDeviceAddressableMemories(PJRT_Device_AddressableMemories_Args* args);
PJRT_Error* QnnDeviceDefaultMemory(PJRT_Device_DefaultMemory_Args* args);

PJRT_Error* QnnMemoryId(PJRT_Memory_Id_Args* args);
PJRT_Error* QnnMemoryKind(PJRT_Memory_Kind_Args* args);
PJRT_Error* QnnMemoryKindId(PJRT_Memory_Kind_Id_Args* args);
PJRT_Error* QnnMemoryDebugString(PJRT_Memory_DebugString_Args* args);
PJRT_Error* QnnMemoryToString(PJRT_Memory_ToString_Args* args);
PJRT_Error* QnnMemoryAddressableByDevices(PJRT_Memory_AddressableByDevices_Args* args);

// Buffer & Event (from qnn_buffer.cc)
PJRT_Error* QnnBufferDestroy(PJRT_Buffer_Destroy_Args* args);
PJRT_Error* QnnBufferElementType(PJRT_Buffer_ElementType_Args* args);
PJRT_Error* QnnBufferDimensions(PJRT_Buffer_Dimensions_Args* args);
PJRT_Error* QnnBufferToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args);
PJRT_Error* QnnBufferOnDeviceSizeInBytes(PJRT_Buffer_OnDeviceSizeInBytes_Args* args);
PJRT_Error* QnnBufferDelete(PJRT_Buffer_Delete_Args* args);
PJRT_Error* QnnBufferIsDeleted(PJRT_Buffer_IsDeleted_Args* args);
PJRT_Error* QnnBufferDevice(PJRT_Buffer_Device_Args* args);
PJRT_Error* QnnBufferMemory(PJRT_Buffer_Memory_Args* args);
PJRT_Error* QnnBufferReadyEvent(PJRT_Buffer_ReadyEvent_Args* args);
PJRT_Error* QnnBufferIsOnCpu(PJRT_Buffer_IsOnCpu_Args* args);
PJRT_Error* QnnBufferCopyToDevice(PJRT_Buffer_CopyToDevice_Args* args);

PJRT_Error* QnnEventDestroy(PJRT_Event_Destroy_Args* args);
PJRT_Error* QnnEventIsReady(PJRT_Event_IsReady_Args* args);
PJRT_Error* QnnEventAwait(PJRT_Event_Await_Args* args);
PJRT_Error* QnnEventOnReady(PJRT_Event_OnReady_Args* args);

// Executable (from qnn_executable.cc)
PJRT_Error* QnnExecutableName(PJRT_Executable_Name_Args* args);
PJRT_Error* QnnExecutableNumOutputs(PJRT_Executable_NumOutputs_Args* args);
PJRT_Error* QnnExecutableDestroy(PJRT_Executable_Destroy_Args* args);
PJRT_Error* QnnLoadedExecutableDestroy(PJRT_LoadedExecutable_Destroy_Args* args);
PJRT_Error* QnnLoadedExecutableGetExecutable(PJRT_LoadedExecutable_GetExecutable_Args* args);
PJRT_Error* QnnLoadedExecutableAddressableDevices(PJRT_LoadedExecutable_AddressableDevices_Args* args);
PJRT_Error* QnnLoadedExecutableDelete(PJRT_LoadedExecutable_Delete_Args* args);
PJRT_Error* QnnLoadedExecutableIsDeleted(PJRT_LoadedExecutable_IsDeleted_Args* args);
PJRT_Error* QnnLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args* args);

// Topology (Optional stubs)
static PJRT_Error* QnnTopologyCreate(PJRT_TopologyDescription_Create_Args* /*args*/) {
  return MakeError(PJRT_Error_Code_UNIMPLEMENTED, "Topology creation not implemented for QNN");
}

static PJRT_Error* QnnTopologyDestroy(PJRT_TopologyDescription_Destroy_Args* /*args*/) {
  return nullptr;
}

static PJRT_Error* QnnTopologyPlatformName(PJRT_TopologyDescription_PlatformName_Args* args) {
  static const char name[] = "qnn";
  args->platform_name = name;
  args->platform_name_size = sizeof(name) - 1;
  return nullptr;
}

}  // namespace jax_qnn

// =============================================================================
// Exported GetPjRtApi Symbol
// =============================================================================

extern "C" {

PJRT_PLUGIN_EXPORT const PJRT_Api* GetPjRtApi() {
  static PJRT_Api api;
  static bool initialized = false;

  if (!initialized) {
    std::memset(&api, 0, sizeof(api));
    api.struct_size = sizeof(PJRT_Api);
    api.extension_start = nullptr;
    api.pjrt_api_version_major = PJRT_API_MAJOR;
    api.pjrt_api_version_minor = PJRT_API_MINOR;

    // Error
    api.PJRT_Error_Destroy = jax_qnn::QnnErrorDestroy;
    api.PJRT_Error_Message = jax_qnn::QnnErrorMessage;
    api.PJRT_Error_GetCode = jax_qnn::QnnErrorGetCode;

    // Plugin
    api.PJRT_Plugin_Initialize = jax_qnn::QnnPluginInitialize;
    api.PJRT_Plugin_Attributes = jax_qnn::QnnPluginAttributes;

    // Client
    api.PJRT_Client_Create = jax_qnn::QnnClientCreate;
    api.PJRT_Client_Destroy = jax_qnn::QnnClientDestroy;
    api.PJRT_Client_PlatformName = jax_qnn::QnnClientPlatformName;
    api.PJRT_Client_ProcessIndex = jax_qnn::QnnClientProcessIndex;
    api.PJRT_Client_PlatformVersion = jax_qnn::QnnClientPlatformVersion;
    api.PJRT_Client_Devices = jax_qnn::QnnClientDevices;
    api.PJRT_Client_AddressableDevices = jax_qnn::QnnClientAddressableDevices;
    api.PJRT_Client_LookupDevice = jax_qnn::QnnClientLookupDevice;
    api.PJRT_Client_LookupAddressableDevice = jax_qnn::QnnClientLookupAddressableDevice;
    api.PJRT_Client_Compile = jax_qnn::QnnClientCompile;
    api.PJRT_Client_BufferFromHostBuffer = jax_qnn::QnnClientBufferFromHostBuffer;

    // Device Description
    api.PJRT_DeviceDescription_Id = jax_qnn::QnnDeviceDescriptionId;
    api.PJRT_DeviceDescription_ProcessIndex = jax_qnn::QnnDeviceDescriptionProcessIndex;
    api.PJRT_DeviceDescription_Attributes = jax_qnn::QnnDeviceDescriptionAttributes;
    api.PJRT_DeviceDescription_Kind = jax_qnn::QnnDeviceDescriptionKind;
    api.PJRT_DeviceDescription_DebugString = jax_qnn::QnnDeviceDescriptionDebugString;
    api.PJRT_DeviceDescription_ToString = jax_qnn::QnnDeviceDescriptionToString;

    // Device
    api.PJRT_Device_GetDescription = jax_qnn::QnnDeviceGetDescription;
    api.PJRT_Device_IsAddressable = jax_qnn::QnnDeviceIsAddressable;
    api.PJRT_Device_LocalHardwareId = jax_qnn::QnnDeviceLocalHardwareId;
    api.PJRT_Device_AddressableMemories = jax_qnn::QnnDeviceAddressableMemories;
    api.PJRT_Device_DefaultMemory = jax_qnn::QnnDeviceDefaultMemory;

    // Memory
    api.PJRT_Memory_Id = jax_qnn::QnnMemoryId;
    api.PJRT_Memory_Kind = jax_qnn::QnnMemoryKind;
    api.PJRT_Memory_Kind_Id = jax_qnn::QnnMemoryKindId;
    api.PJRT_Memory_DebugString = jax_qnn::QnnMemoryDebugString;
    api.PJRT_Memory_ToString = jax_qnn::QnnMemoryToString;
    api.PJRT_Memory_AddressableByDevices = jax_qnn::QnnMemoryAddressableByDevices;

    // Buffer
    api.PJRT_Buffer_Destroy = jax_qnn::QnnBufferDestroy;
    api.PJRT_Buffer_ElementType = jax_qnn::QnnBufferElementType;
    api.PJRT_Buffer_Dimensions = jax_qnn::QnnBufferDimensions;
    api.PJRT_Buffer_ToHostBuffer = jax_qnn::QnnBufferToHostBuffer;
    api.PJRT_Buffer_OnDeviceSizeInBytes = jax_qnn::QnnBufferOnDeviceSizeInBytes;
    api.PJRT_Buffer_Delete = jax_qnn::QnnBufferDelete;
    api.PJRT_Buffer_IsDeleted = jax_qnn::QnnBufferIsDeleted;
    api.PJRT_Buffer_Device = jax_qnn::QnnBufferDevice;
    api.PJRT_Buffer_Memory = jax_qnn::QnnBufferMemory;
    api.PJRT_Buffer_ReadyEvent = jax_qnn::QnnBufferReadyEvent;
    api.PJRT_Buffer_IsOnCpu = jax_qnn::QnnBufferIsOnCpu;
    api.PJRT_Buffer_CopyToDevice = jax_qnn::QnnBufferCopyToDevice;

    // Executable
    api.PJRT_Executable_Name = jax_qnn::QnnExecutableName;
    api.PJRT_Executable_NumOutputs = jax_qnn::QnnExecutableNumOutputs;
    api.PJRT_Executable_Destroy = jax_qnn::QnnExecutableDestroy;
    api.PJRT_LoadedExecutable_Destroy = jax_qnn::QnnLoadedExecutableDestroy;
    api.PJRT_LoadedExecutable_GetExecutable = jax_qnn::QnnLoadedExecutableGetExecutable;
    api.PJRT_LoadedExecutable_AddressableDevices = jax_qnn::QnnLoadedExecutableAddressableDevices;
    api.PJRT_LoadedExecutable_Delete = jax_qnn::QnnLoadedExecutableDelete;
    api.PJRT_LoadedExecutable_IsDeleted = jax_qnn::QnnLoadedExecutableIsDeleted;
    api.PJRT_LoadedExecutable_Execute = jax_qnn::QnnLoadedExecutableExecute;

    // Event
    api.PJRT_Event_Destroy = jax_qnn::QnnEventDestroy;
    api.PJRT_Event_IsReady = jax_qnn::QnnEventIsReady;
    api.PJRT_Event_Await = jax_qnn::QnnEventAwait;
    api.PJRT_Event_OnReady = jax_qnn::QnnEventOnReady;

    // Topology
    api.PJRT_TopologyDescription_Create = jax_qnn::QnnTopologyCreate;
    api.PJRT_TopologyDescription_Destroy = jax_qnn::QnnTopologyDestroy;
    api.PJRT_TopologyDescription_PlatformName = jax_qnn::QnnTopologyPlatformName;

    initialized = true;
  }

  return &api;
}

}  // extern "C"
