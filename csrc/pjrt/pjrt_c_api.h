/* Vendored minimal PJRT C API header for JAX-QNN plugin.
 *
 * This is a self-contained subset of the PJRT C API from OpenXLA/XLA,
 * containing only the types and function signatures needed to implement
 * a minimal PJRT plugin.
 *
 * Source: https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h
 * License: Apache 2.0
 *
 * For the full PJRT C API, see the OpenXLA repository.
 */

#ifndef PJRT_C_API_H_
#define PJRT_C_API_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define PJRT_STRUCT_SIZE(struct_type, last_field) \
  (offsetof(struct_type, last_field) + sizeof(((struct_type*)0)->last_field))

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Forward declarations
// =============================================================================

typedef struct PJRT_Api PJRT_Api;
typedef struct PJRT_Error PJRT_Error;
typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_Memory PJRT_Memory;
typedef struct PJRT_Buffer PJRT_Buffer;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Executable PJRT_Executable;
typedef struct PJRT_Event PJRT_Event;
typedef struct PJRT_DeviceTopology PJRT_DeviceTopology;
typedef struct PJRT_DeviceDescription PJRT_DeviceDescription;

// =============================================================================
// Extensions
// =============================================================================

typedef struct PJRT_Extension_Base {
  size_t struct_size;
  int32_t type;
  struct PJRT_Extension_Base* next;
} PJRT_Extension_Base;

// =============================================================================
// Named values (key-value pairs for options)
// =============================================================================

typedef enum {
  PJRT_NamedValue_kString = 0,
  PJRT_NamedValue_kInt64,
  PJRT_NamedValue_kInt64List,
  PJRT_NamedValue_kFloat,
  PJRT_NamedValue_kBool,
} PJRT_NamedValue_Type;

typedef struct PJRT_NamedValue {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* name;
  size_t name_size;
  PJRT_NamedValue_Type type;
  union {
    const char* string_value;
    int64_t int64_value;
    const int64_t* int64_array_value;
    float float_value;
    bool bool_value;
  };
  size_t value_size;  // For string: length. For int64_array: num elements.
} PJRT_NamedValue;

// =============================================================================
// Error
// =============================================================================

typedef enum {
  PJRT_Error_Code_CANCELLED = 1,
  PJRT_Error_Code_UNKNOWN = 2,
  PJRT_Error_Code_INVALID_ARGUMENT = 3,
  PJRT_Error_Code_DEADLINE_EXCEEDED = 4,
  PJRT_Error_Code_NOT_FOUND = 5,
  PJRT_Error_Code_ALREADY_EXISTS = 6,
  PJRT_Error_Code_PERMISSION_DENIED = 7,
  PJRT_Error_Code_RESOURCE_EXHAUSTED = 8,
  PJRT_Error_Code_FAILED_PRECONDITION = 9,
  PJRT_Error_Code_ABORTED = 10,
  PJRT_Error_Code_OUT_OF_RANGE = 11,
  PJRT_Error_Code_UNIMPLEMENTED = 12,
  PJRT_Error_Code_INTERNAL = 13,
  PJRT_Error_Code_UNAVAILABLE = 14,
  PJRT_Error_Code_DATA_LOSS = 15,
  PJRT_Error_Code_UNAUTHENTICATED = 16,
} PJRT_Error_Code;

struct PJRT_Error_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Error* error;
};
typedef PJRT_Error* PJRT_Error_Destroy_Fn(struct PJRT_Error_Destroy_Args* args);

struct PJRT_Error_Message_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Error* error;
  const char* message;      // out
  size_t message_size;       // out
};
typedef PJRT_Error* PJRT_Error_Message_Fn(struct PJRT_Error_Message_Args* args);

struct PJRT_Error_GetCode_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Error* error;
  PJRT_Error_Code code;  // out
};
typedef PJRT_Error* PJRT_Error_GetCode_Fn(struct PJRT_Error_GetCode_Args* args);

// =============================================================================
// Plugin initialization
// =============================================================================

struct PJRT_Plugin_Initialize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
};
typedef PJRT_Error* PJRT_Plugin_Initialize_Fn(
    struct PJRT_Plugin_Initialize_Args* args);

struct PJRT_Plugin_Attributes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  size_t num_attributes;        // out
  const PJRT_NamedValue* attributes;  // out
};
typedef PJRT_Error* PJRT_Plugin_Attributes_Fn(
    struct PJRT_Plugin_Attributes_Args* args);

// =============================================================================
// Client
// =============================================================================

// Key-value store callbacks (used for distributed setup)
typedef void (*PJRT_KeyValueGetCallback)(
    const char* key, size_t key_size, 
    void* value, size_t value_size, void* user_arg);
typedef void (*PJRT_KeyValuePutCallback)(
    const char* key, size_t key_size, 
    const char* value, size_t value_size, void* user_arg);

struct PJRT_Client_Create_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;                    // out
  const PJRT_NamedValue* create_options;
  size_t num_options;
  // KV store callbacks (may be null for single-process)
  PJRT_KeyValueGetCallback kv_get_callback;
  void* kv_get_user_arg;
  PJRT_KeyValuePutCallback kv_put_callback;
  void* kv_put_user_arg;
};
typedef PJRT_Error* PJRT_Client_Create_Fn(struct PJRT_Client_Create_Args* args);

struct PJRT_Client_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
};
typedef PJRT_Error* PJRT_Client_Destroy_Fn(struct PJRT_Client_Destroy_Args* args);

struct PJRT_Client_PlatformName_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const char* platform_name;    // out
  size_t platform_name_size;     // out
};
typedef PJRT_Error* PJRT_Client_PlatformName_Fn(
    struct PJRT_Client_PlatformName_Args* args);

struct PJRT_Client_ProcessIndex_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  int process_index;  // out
};
typedef PJRT_Error* PJRT_Client_ProcessIndex_Fn(
    struct PJRT_Client_ProcessIndex_Args* args);

struct PJRT_Client_PlatformVersion_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const char* platform_version;    // out
  size_t platform_version_size;     // out
};
typedef PJRT_Error* PJRT_Client_PlatformVersion_Fn(
    struct PJRT_Client_PlatformVersion_Args* args);

struct PJRT_Client_Devices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_Device* const* devices;  // out
  size_t num_devices;             // out
};
typedef PJRT_Error* PJRT_Client_Devices_Fn(
    struct PJRT_Client_Devices_Args* args);

struct PJRT_Client_AddressableDevices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_Device* const* addressable_devices;  // out
  size_t num_addressable_devices;             // out
};
typedef PJRT_Error* PJRT_Client_AddressableDevices_Fn(
    struct PJRT_Client_AddressableDevices_Args* args);

struct PJRT_Client_LookupDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  int id;
  PJRT_Device* device;  // out
};
typedef PJRT_Error* PJRT_Client_LookupDevice_Fn(
    struct PJRT_Client_LookupDevice_Args* args);

struct PJRT_Client_LookupAddressableDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  int local_hardware_id;
  PJRT_Device* addressable_device;  // out
};
typedef PJRT_Error* PJRT_Client_LookupAddressableDevice_Fn(
    struct PJRT_Client_LookupAddressableDevice_Args* args);

// =============================================================================
// Compile (Client → Executable)
// =============================================================================

typedef enum {
  PJRT_Program_Format_MLIR = 0,
  PJRT_Program_Format_HLO = 1,
} PJRT_Program_Format;

struct PJRT_Program {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* format;
  size_t format_size;
  const char* code;
  size_t code_size;
};

struct PJRT_Client_Compile_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const struct PJRT_Program* program;
  // compile_options is serialized CompileOptionsProto
  const char* compile_options;
  size_t compile_options_size;
  PJRT_LoadedExecutable* executable;  // out
};
typedef PJRT_Error* PJRT_Client_Compile_Fn(
    struct PJRT_Client_Compile_Args* args);

// =============================================================================
// Device
// =============================================================================

struct PJRT_DeviceDescription_Id_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  int id;  // out
};
typedef PJRT_Error* PJRT_DeviceDescription_Id_Fn(
    struct PJRT_DeviceDescription_Id_Args* args);

struct PJRT_DeviceDescription_ProcessIndex_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  int process_index;  // out
};
typedef PJRT_Error* PJRT_DeviceDescription_ProcessIndex_Fn(
    struct PJRT_DeviceDescription_ProcessIndex_Args* args);

struct PJRT_DeviceDescription_Attributes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  size_t num_attributes;            // out
  const PJRT_NamedValue* attributes; // out
};
typedef PJRT_Error* PJRT_DeviceDescription_Attributes_Fn(
    struct PJRT_DeviceDescription_Attributes_Args* args);

struct PJRT_DeviceDescription_Kind_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const char* device_kind;    // out
  size_t device_kind_size;     // out
};
typedef PJRT_Error* PJRT_DeviceDescription_Kind_Fn(
    struct PJRT_DeviceDescription_Kind_Args* args);

struct PJRT_DeviceDescription_DebugString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const char* debug_string;    // out
  size_t debug_string_size;     // out
};
typedef PJRT_Error* PJRT_DeviceDescription_DebugString_Fn(
    struct PJRT_DeviceDescription_DebugString_Args* args);

struct PJRT_DeviceDescription_ToString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const char* to_string;    // out
  size_t to_string_size;     // out
};
typedef PJRT_Error* PJRT_DeviceDescription_ToString_Fn(
    struct PJRT_DeviceDescription_ToString_Args* args);

struct PJRT_Device_GetDescription_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  PJRT_DeviceDescription* device_description;  // out
};
typedef PJRT_Error* PJRT_Device_GetDescription_Fn(
    struct PJRT_Device_GetDescription_Args* args);

struct PJRT_Device_IsAddressable_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  bool is_addressable;  // out
};
typedef PJRT_Error* PJRT_Device_IsAddressable_Fn(
    struct PJRT_Device_IsAddressable_Args* args);

struct PJRT_Device_LocalHardwareId_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  int local_hardware_id;  // out
};
typedef PJRT_Error* PJRT_Device_LocalHardwareId_Fn(
    struct PJRT_Device_LocalHardwareId_Args* args);

struct PJRT_Device_AddressableMemories_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  PJRT_Memory* const* memories;  // out
  size_t num_memories;             // out
};
typedef PJRT_Error* PJRT_Device_AddressableMemories_Fn(
    struct PJRT_Device_AddressableMemories_Args* args);

struct PJRT_Device_DefaultMemory_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  PJRT_Memory* memory;  // out
};
typedef PJRT_Error* PJRT_Device_DefaultMemory_Fn(
    struct PJRT_Device_DefaultMemory_Args* args);

// =============================================================================
// Memory
// =============================================================================

struct PJRT_Memory_Id_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  int id;  // out
};
typedef PJRT_Error* PJRT_Memory_Id_Fn(struct PJRT_Memory_Id_Args* args);

struct PJRT_Memory_Kind_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  const char* kind;    // out
  size_t kind_size;     // out
};
typedef PJRT_Error* PJRT_Memory_Kind_Fn(struct PJRT_Memory_Kind_Args* args);

struct PJRT_Memory_Kind_Id_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  int kind_id;  // out
};
typedef PJRT_Error* PJRT_Memory_Kind_Id_Fn(struct PJRT_Memory_Kind_Id_Args* args);

struct PJRT_Memory_DebugString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  const char* debug_string;    // out
  size_t debug_string_size;     // out
};
typedef PJRT_Error* PJRT_Memory_DebugString_Fn(
    struct PJRT_Memory_DebugString_Args* args);

struct PJRT_Memory_ToString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  const char* to_string;    // out
  size_t to_string_size;     // out
};
typedef PJRT_Error* PJRT_Memory_ToString_Fn(
    struct PJRT_Memory_ToString_Args* args);

struct PJRT_Memory_AddressableByDevices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  PJRT_Device* const* devices;  // out
  size_t num_devices;             // out
};
typedef PJRT_Error* PJRT_Memory_AddressableByDevices_Fn(
    struct PJRT_Memory_AddressableByDevices_Args* args);

// =============================================================================
// Buffer
// =============================================================================

typedef enum {
  PJRT_Buffer_Type_INVALID = 0,
  PJRT_Buffer_Type_PRED = 1,
  PJRT_Buffer_Type_S8 = 2,
  PJRT_Buffer_Type_S16 = 3,
  PJRT_Buffer_Type_S32 = 4,
  PJRT_Buffer_Type_S64 = 5,
  PJRT_Buffer_Type_U8 = 6,
  PJRT_Buffer_Type_U16 = 7,
  PJRT_Buffer_Type_U32 = 8,
  PJRT_Buffer_Type_U64 = 9,
  PJRT_Buffer_Type_F16 = 10,
  PJRT_Buffer_Type_F32 = 11,
  PJRT_Buffer_Type_F64 = 12,
  PJRT_Buffer_Type_BF16 = 16,
  PJRT_Buffer_Type_C64 = 15,
  PJRT_Buffer_Type_C128 = 18,
  PJRT_Buffer_Type_F8E5M2 = 19,
  PJRT_Buffer_Type_F8E4M3FN = 20,
  PJRT_Buffer_Type_F8E4M3B11FNUZ = 23,
  PJRT_Buffer_Type_F8E5M2FNUZ = 24,
  PJRT_Buffer_Type_F8E4M3FNUZ = 25,
} PJRT_Buffer_Type;

typedef enum {
  PJRT_HostBufferSemantics_kImmutableOnlyDuringCall = 0,
  PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes = 1,
  PJRT_HostBufferSemantics_kImmutableZeroCopy = 2,
  PJRT_HostBufferSemantics_kMutableZeroCopy = 3,
} PJRT_HostBufferSemantics;

struct PJRT_Client_BufferFromHostBuffer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const void* data;
  PJRT_Buffer_Type type;
  const int64_t* dims;
  size_t num_dims;
  const int64_t* byte_strides;
  size_t num_byte_strides;
  PJRT_HostBufferSemantics host_buffer_semantics;
  PJRT_Device* device;
  PJRT_Memory* memory;
  // device_layout is optional  
  PJRT_Buffer* buffer;  // out
  PJRT_Event* done_with_host_buffer;  // out, optional
};
typedef PJRT_Error* PJRT_Client_BufferFromHostBuffer_Fn(
    struct PJRT_Client_BufferFromHostBuffer_Args* args);

struct PJRT_Buffer_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
};
typedef PJRT_Error* PJRT_Buffer_Destroy_Fn(
    struct PJRT_Buffer_Destroy_Args* args);

struct PJRT_Buffer_ElementType_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Buffer_Type type;  // out
};
typedef PJRT_Error* PJRT_Buffer_ElementType_Fn(
    struct PJRT_Buffer_ElementType_Args* args);

struct PJRT_Buffer_Dimensions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  const int64_t* dims;  // out
  size_t num_dims;        // out
};
typedef PJRT_Error* PJRT_Buffer_Dimensions_Fn(
    struct PJRT_Buffer_Dimensions_Args* args);

struct PJRT_Buffer_ToHostBuffer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* src;
  void* dst;
  size_t dst_size;
  PJRT_Event* event;  // out
};
typedef PJRT_Error* PJRT_Buffer_ToHostBuffer_Fn(
    struct PJRT_Buffer_ToHostBuffer_Args* args);

struct PJRT_Buffer_OnDeviceSizeInBytes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  size_t on_device_size_in_bytes;  // out
};
typedef PJRT_Error* PJRT_Buffer_OnDeviceSizeInBytes_Fn(
    struct PJRT_Buffer_OnDeviceSizeInBytes_Args* args);

struct PJRT_Buffer_Delete_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
};
typedef PJRT_Error* PJRT_Buffer_Delete_Fn(struct PJRT_Buffer_Delete_Args* args);

struct PJRT_Buffer_IsDeleted_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  bool is_deleted;  // out
};
typedef PJRT_Error* PJRT_Buffer_IsDeleted_Fn(
    struct PJRT_Buffer_IsDeleted_Args* args);

struct PJRT_Buffer_Device_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Device* device;  // out
};
typedef PJRT_Error* PJRT_Buffer_Device_Fn(struct PJRT_Buffer_Device_Args* args);

struct PJRT_Buffer_Memory_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Memory* memory;  // out
};
typedef PJRT_Error* PJRT_Buffer_Memory_Fn(struct PJRT_Buffer_Memory_Args* args);

struct PJRT_Buffer_ReadyEvent_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Event* event;  // out
};
typedef PJRT_Error* PJRT_Buffer_ReadyEvent_Fn(
    struct PJRT_Buffer_ReadyEvent_Args* args);

struct PJRT_Buffer_IsOnCpu_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  bool is_on_cpu;  // out
};
typedef PJRT_Error* PJRT_Buffer_IsOnCpu_Fn(
    struct PJRT_Buffer_IsOnCpu_Args* args);

struct PJRT_Buffer_CopyToDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Device* dst_device;
  PJRT_Buffer* dst_buffer;  // out
};
typedef PJRT_Error* PJRT_Buffer_CopyToDevice_Fn(
    struct PJRT_Buffer_CopyToDevice_Args* args);

// =============================================================================
// Executable
// =============================================================================

struct PJRT_Executable_Name_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  const char* executable_name;  // out
  size_t executable_name_size;   // out
};
typedef PJRT_Error* PJRT_Executable_Name_Fn(
    struct PJRT_Executable_Name_Args* args);

struct PJRT_Executable_NumOutputs_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_outputs;  // out
};
typedef PJRT_Error* PJRT_Executable_NumOutputs_Fn(
    struct PJRT_Executable_NumOutputs_Args* args);

struct PJRT_Executable_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
};
typedef PJRT_Error* PJRT_Executable_Destroy_Fn(
    struct PJRT_Executable_Destroy_Args* args);

struct PJRT_LoadedExecutable_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
};
typedef PJRT_Error* PJRT_LoadedExecutable_Destroy_Fn(
    struct PJRT_LoadedExecutable_Destroy_Args* args);

struct PJRT_LoadedExecutable_GetExecutable_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* loaded_executable;
  PJRT_Executable* executable;  // out
};
typedef PJRT_Error* PJRT_LoadedExecutable_GetExecutable_Fn(
    struct PJRT_LoadedExecutable_GetExecutable_Args* args);

struct PJRT_LoadedExecutable_AddressableDevices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  PJRT_Device* const* addressable_devices;  // out
  size_t num_addressable_devices;             // out
};
typedef PJRT_Error* PJRT_LoadedExecutable_AddressableDevices_Fn(
    struct PJRT_LoadedExecutable_AddressableDevices_Args* args);

struct PJRT_LoadedExecutable_Delete_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
};
typedef PJRT_Error* PJRT_LoadedExecutable_Delete_Fn(
    struct PJRT_LoadedExecutable_Delete_Args* args);

struct PJRT_LoadedExecutable_IsDeleted_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  bool is_deleted;  // out
};
typedef PJRT_Error* PJRT_LoadedExecutable_IsDeleted_Fn(
    struct PJRT_LoadedExecutable_IsDeleted_Args* args);

// Execute
typedef struct PJRT_ExecuteOptions {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  // These are optional fields for controlling execution.
  int32_t launch_id;
  bool non_donatable_input_indices_is_set;
  const int64_t* non_donatable_input_indices;
  size_t num_non_donatable_input_indices;
} PJRT_ExecuteOptions;

struct PJRT_LoadedExecutable_Execute_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  PJRT_ExecuteOptions* options;
  // Argument buffers: argument_lists[i][j] = j-th arg for i-th device
  PJRT_Buffer* const* const* argument_lists;
  size_t num_devices;
  size_t num_args;
  // Output buffers: output_lists[i][j] = j-th output for i-th device
  PJRT_Buffer** const* output_lists;
  // If this field is non-null, the execution is non-blocking.
  PJRT_Event** device_complete_events;  // optional, out
  bool execute_device;
};
typedef PJRT_Error* PJRT_LoadedExecutable_Execute_Fn(
    struct PJRT_LoadedExecutable_Execute_Args* args);

// =============================================================================
// Event
// =============================================================================

struct PJRT_Event_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Event* event;
};
typedef PJRT_Error* PJRT_Event_Destroy_Fn(struct PJRT_Event_Destroy_Args* args);

struct PJRT_Event_IsReady_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Event* event;
  bool is_ready;  // out
};
typedef PJRT_Error* PJRT_Event_IsReady_Fn(struct PJRT_Event_IsReady_Args* args);

struct PJRT_Event_Await_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Event* event;
};
typedef PJRT_Error* PJRT_Event_Await_Fn(struct PJRT_Event_Await_Args* args);

typedef void (*PJRT_Event_OnReadyCallback)(PJRT_Error* error, void* user_arg);
struct PJRT_Event_OnReady_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Event* event;
  PJRT_Event_OnReadyCallback callback;
  void* user_arg;
};
typedef PJRT_Error* PJRT_Event_OnReady_Fn(struct PJRT_Event_OnReady_Args* args);

// =============================================================================
// TopologyDescription (optional)
// =============================================================================

struct PJRT_TopologyDescription_Create_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceTopology* topology;  // out
  const PJRT_NamedValue* create_options;
  size_t num_options;
};
typedef PJRT_Error* PJRT_TopologyDescription_Create_Fn(
    struct PJRT_TopologyDescription_Create_Args* args);

struct PJRT_TopologyDescription_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceTopology* topology;
};
typedef PJRT_Error* PJRT_TopologyDescription_Destroy_Fn(
    struct PJRT_TopologyDescription_Destroy_Args* args);

struct PJRT_TopologyDescription_PlatformName_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceTopology* topology;
  const char* platform_name;    // out
  size_t platform_name_size;     // out
};
typedef PJRT_Error* PJRT_TopologyDescription_PlatformName_Fn(
    struct PJRT_TopologyDescription_PlatformName_Args* args);

// =============================================================================
// The PJRT_Api struct — the main dispatch table
// =============================================================================

// API version
#define PJRT_API_MAJOR 0
#define PJRT_API_MINOR 56

struct PJRT_Api {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;

  size_t pjrt_api_version_major;
  size_t pjrt_api_version_minor;

  // Error
  PJRT_Error_Destroy_Fn* PJRT_Error_Destroy;
  PJRT_Error_Message_Fn* PJRT_Error_Message;
  PJRT_Error_GetCode_Fn* PJRT_Error_GetCode;

  // Plugin
  PJRT_Plugin_Initialize_Fn* PJRT_Plugin_Initialize;
  PJRT_Plugin_Attributes_Fn* PJRT_Plugin_Attributes;

  // Client
  PJRT_Client_Create_Fn* PJRT_Client_Create;
  PJRT_Client_Destroy_Fn* PJRT_Client_Destroy;
  PJRT_Client_PlatformName_Fn* PJRT_Client_PlatformName;
  PJRT_Client_ProcessIndex_Fn* PJRT_Client_ProcessIndex;
  PJRT_Client_PlatformVersion_Fn* PJRT_Client_PlatformVersion;
  PJRT_Client_Devices_Fn* PJRT_Client_Devices;
  PJRT_Client_AddressableDevices_Fn* PJRT_Client_AddressableDevices;
  PJRT_Client_LookupDevice_Fn* PJRT_Client_LookupDevice;
  PJRT_Client_LookupAddressableDevice_Fn* PJRT_Client_LookupAddressableDevice;
  PJRT_Client_Compile_Fn* PJRT_Client_Compile;
  PJRT_Client_BufferFromHostBuffer_Fn* PJRT_Client_BufferFromHostBuffer;

  // Device
  PJRT_DeviceDescription_Id_Fn* PJRT_DeviceDescription_Id;
  PJRT_DeviceDescription_ProcessIndex_Fn* PJRT_DeviceDescription_ProcessIndex;
  PJRT_DeviceDescription_Attributes_Fn* PJRT_DeviceDescription_Attributes;
  PJRT_DeviceDescription_Kind_Fn* PJRT_DeviceDescription_Kind;
  PJRT_DeviceDescription_DebugString_Fn* PJRT_DeviceDescription_DebugString;
  PJRT_DeviceDescription_ToString_Fn* PJRT_DeviceDescription_ToString;
  PJRT_Device_GetDescription_Fn* PJRT_Device_GetDescription;
  PJRT_Device_IsAddressable_Fn* PJRT_Device_IsAddressable;
  PJRT_Device_LocalHardwareId_Fn* PJRT_Device_LocalHardwareId;
  PJRT_Device_AddressableMemories_Fn* PJRT_Device_AddressableMemories;
  PJRT_Device_DefaultMemory_Fn* PJRT_Device_DefaultMemory;

  // Memory
  PJRT_Memory_Id_Fn* PJRT_Memory_Id;
  PJRT_Memory_Kind_Fn* PJRT_Memory_Kind;
  PJRT_Memory_Kind_Id_Fn* PJRT_Memory_Kind_Id;
  PJRT_Memory_DebugString_Fn* PJRT_Memory_DebugString;
  PJRT_Memory_ToString_Fn* PJRT_Memory_ToString;
  PJRT_Memory_AddressableByDevices_Fn* PJRT_Memory_AddressableByDevices;

  // Buffer
  PJRT_Buffer_Destroy_Fn* PJRT_Buffer_Destroy;
  PJRT_Buffer_ElementType_Fn* PJRT_Buffer_ElementType;
  PJRT_Buffer_Dimensions_Fn* PJRT_Buffer_Dimensions;
  PJRT_Buffer_ToHostBuffer_Fn* PJRT_Buffer_ToHostBuffer;
  PJRT_Buffer_OnDeviceSizeInBytes_Fn* PJRT_Buffer_OnDeviceSizeInBytes;
  PJRT_Buffer_Delete_Fn* PJRT_Buffer_Delete;
  PJRT_Buffer_IsDeleted_Fn* PJRT_Buffer_IsDeleted;
  PJRT_Buffer_Device_Fn* PJRT_Buffer_Device;
  PJRT_Buffer_Memory_Fn* PJRT_Buffer_Memory;
  PJRT_Buffer_ReadyEvent_Fn* PJRT_Buffer_ReadyEvent;
  PJRT_Buffer_IsOnCpu_Fn* PJRT_Buffer_IsOnCpu;
  PJRT_Buffer_CopyToDevice_Fn* PJRT_Buffer_CopyToDevice;

  // Executable
  PJRT_Executable_Name_Fn* PJRT_Executable_Name;
  PJRT_Executable_NumOutputs_Fn* PJRT_Executable_NumOutputs;
  PJRT_Executable_Destroy_Fn* PJRT_Executable_Destroy;
  PJRT_LoadedExecutable_Destroy_Fn* PJRT_LoadedExecutable_Destroy;
  PJRT_LoadedExecutable_GetExecutable_Fn* PJRT_LoadedExecutable_GetExecutable;
  PJRT_LoadedExecutable_AddressableDevices_Fn* PJRT_LoadedExecutable_AddressableDevices;
  PJRT_LoadedExecutable_Delete_Fn* PJRT_LoadedExecutable_Delete;
  PJRT_LoadedExecutable_IsDeleted_Fn* PJRT_LoadedExecutable_IsDeleted;
  PJRT_LoadedExecutable_Execute_Fn* PJRT_LoadedExecutable_Execute;

  // Event
  PJRT_Event_Destroy_Fn* PJRT_Event_Destroy;
  PJRT_Event_IsReady_Fn* PJRT_Event_IsReady;
  PJRT_Event_Await_Fn* PJRT_Event_Await;
  PJRT_Event_OnReady_Fn* PJRT_Event_OnReady;

  // Topology
  PJRT_TopologyDescription_Create_Fn* PJRT_TopologyDescription_Create;
  PJRT_TopologyDescription_Destroy_Fn* PJRT_TopologyDescription_Destroy;
  PJRT_TopologyDescription_PlatformName_Fn* PJRT_TopologyDescription_PlatformName;
};

// =============================================================================
// GetPjRtApi — the entry point that plugins must export
// =============================================================================

#ifdef _WIN32
  #ifdef JAX_QNN_BUILDING_DLL
    #define PJRT_PLUGIN_EXPORT __declspec(dllexport)
  #else
    #define PJRT_PLUGIN_EXPORT __declspec(dllimport)
  #endif
#else
  #define PJRT_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

PJRT_PLUGIN_EXPORT const PJRT_Api* GetPjRtApi();

#ifdef __cplusplus
}
#endif

#endif  // PJRT_C_API_H_
