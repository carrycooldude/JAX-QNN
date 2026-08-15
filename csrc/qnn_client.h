// =============================================================================
// qnn_client.h — PJRT Client implementation for QNN
// =============================================================================

#ifndef JAX_QNN_QNN_CLIENT_H_
#define JAX_QNN_QNN_CLIENT_H_

#include "pjrt/pjrt_c_api.h"

namespace jax_qnn {

// PJRT Client API implementations
PJRT_Error* QnnClientCreate(PJRT_Client_Create_Args* args);
PJRT_Error* QnnClientDestroy(PJRT_Client_Destroy_Args* args);
PJRT_Error* QnnClientPlatformName(PJRT_Client_PlatformName_Args* args);
PJRT_Error* QnnClientProcessIndex(PJRT_Client_ProcessIndex_Args* args);
PJRT_Error* QnnClientPlatformVersion(PJRT_Client_PlatformVersion_Args* args);
PJRT_Error* QnnClientDevices(PJRT_Client_Devices_Args* args);
PJRT_Error* QnnClientAddressableDevices(PJRT_Client_AddressableDevices_Args* args);
PJRT_Error* QnnClientLookupDevice(PJRT_Client_LookupDevice_Args* args);
PJRT_Error* QnnClientLookupAddressableDevice(PJRT_Client_LookupAddressableDevice_Args* args);
PJRT_Error* QnnClientCompile(PJRT_Client_Compile_Args* args);
PJRT_Error* QnnClientBufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args);

}  // namespace jax_qnn

#endif  // JAX_QNN_QNN_CLIENT_H_
