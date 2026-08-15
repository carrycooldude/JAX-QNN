// =============================================================================
// qnn_buffer.cc — PJRT Buffer implementation for QNN
// =============================================================================
//
// Handles buffer lifecycle, host↔device transfers, and buffer queries.
// Since QNN on Snapdragon uses shared memory (CPU and NPU access the same
// physical memory), the "transfer" is essentially a memcpy or even zero-copy.
//
// =============================================================================

#include "qnn_types.h"
#include "pjrt/pjrt_c_api.h"

#include <cstring>
#include <iostream>

namespace jax_qnn {

// =============================================================================
// Buffer API
// =============================================================================

PJRT_Error* QnnBufferDestroy(PJRT_Buffer_Destroy_Args* args) {
  auto* buf = reinterpret_cast<QnnBufferState*>(args->buffer);
  delete buf;
  return NoError();
}

PJRT_Error* QnnBufferElementType(PJRT_Buffer_ElementType_Args* args) {
  auto* buf = reinterpret_cast<QnnBufferState*>(args->buffer);
  args->type = buf->element_type;
  return NoError();
}

PJRT_Error* QnnBufferDimensions(PJRT_Buffer_Dimensions_Args* args) {
  auto* buf = reinterpret_cast<QnnBufferState*>(args->buffer);
  args->dims = buf->dimensions.data();
  args->num_dims = buf->dimensions.size();
  return NoError();
}

PJRT_Error* QnnBufferToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
  auto* buf = reinterpret_cast<QnnBufferState*>(args->src);
  
  if (args->dst && args->dst_size > 0) {
    size_t copy_size = std::min(args->dst_size, buf->data.size());
    std::memcpy(args->dst, buf->data.data(), copy_size);
  }
  
  // Create a ready event
  auto* event = new QnnEventState();
  event->is_ready.store(true);
  args->event = reinterpret_cast<PJRT_Event*>(event);
  
  return NoError();
}

PJRT_Error* QnnBufferOnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
  auto* buf = reinterpret_cast<QnnBufferState*>(args->buffer);
  args->on_device_size_in_bytes = buf->GetByteSize();
  return NoError();
}

PJRT_Error* QnnBufferDelete(PJRT_Buffer_Delete_Args* args) {
  auto* buf = reinterpret_cast<QnnBufferState*>(args->buffer);
  buf->is_deleted = true;
  buf->data.clear();
  return NoError();
}

PJRT_Error* QnnBufferIsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
  auto* buf = reinterpret_cast<QnnBufferState*>(args->buffer);
  args->is_deleted = buf->is_deleted;
  return NoError();
}

PJRT_Error* QnnBufferDevice(PJRT_Buffer_Device_Args* args) {
  auto* buf = reinterpret_cast<QnnBufferState*>(args->buffer);
  args->device = reinterpret_cast<PJRT_Device*>(buf->device);
  return NoError();
}

PJRT_Error* QnnBufferMemory(PJRT_Buffer_Memory_Args* args) {
  auto* buf = reinterpret_cast<QnnBufferState*>(args->buffer);
  args->memory = reinterpret_cast<PJRT_Memory*>(buf->memory);
  return NoError();
}

PJRT_Error* QnnBufferReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
  // Buffer is always ready in synchronous mode
  auto* event = new QnnEventState();
  event->is_ready.store(true);
  args->event = reinterpret_cast<PJRT_Event*>(event);
  return NoError();
}

PJRT_Error* QnnBufferIsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
  // QNN buffers are on shared memory, accessible by CPU
  // But we report false to indicate it's a "device" buffer
  args->is_on_cpu = false;
  return NoError();
}

PJRT_Error* QnnBufferCopyToDevice(PJRT_Buffer_CopyToDevice_Args* args) {
  auto* src = reinterpret_cast<QnnBufferState*>(args->buffer);
  
  // Create a copy of the buffer
  auto* dst = new QnnBufferState();
  dst->client = src->client;
  dst->device = src->device;  // Same device for now
  dst->memory = src->memory;
  dst->element_type = src->element_type;
  dst->dimensions = src->dimensions;
  dst->data = src->data;
  dst->is_ready.store(true);
  
  args->dst_buffer = reinterpret_cast<PJRT_Buffer*>(dst);
  return NoError();
}

// =============================================================================
// Event API
// =============================================================================

PJRT_Error* QnnEventDestroy(PJRT_Event_Destroy_Args* args) {
  auto* event = reinterpret_cast<QnnEventState*>(args->event);
  delete event;
  return NoError();
}

PJRT_Error* QnnEventIsReady(PJRT_Event_IsReady_Args* args) {
  auto* event = reinterpret_cast<QnnEventState*>(args->event);
  args->is_ready = event->is_ready.load();
  return NoError();
}

PJRT_Error* QnnEventAwait(PJRT_Event_Await_Args* args) {
  auto* event = reinterpret_cast<QnnEventState*>(args->event);
  // In synchronous mode, events are always ready
  while (!event->is_ready.load()) {
    // Spin wait — in production, use condition variable
  }
  return event->error;
}

PJRT_Error* QnnEventOnReady(PJRT_Event_OnReady_Args* args) {
  auto* event = reinterpret_cast<QnnEventState*>(args->event);
  if (event->is_ready.load()) {
    // Already ready, call immediately
    args->callback(event->error, args->user_arg);
  } else {
    std::lock_guard<std::mutex> lock(event->callback_mutex);
    event->callbacks.emplace_back(args->callback, args->user_arg);
  }
  return NoError();
}

}  // namespace jax_qnn
