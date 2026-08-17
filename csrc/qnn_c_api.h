// =============================================================================
// qnn_c_api.h — Direct C ABI exports for Python ctypes bridge
// =============================================================================

#ifndef JAX_QNN_C_API_H_
#define JAX_QNN_C_API_H_

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#define JAX_QNN_EXPORT __declspec(dllexport)
#else
#define JAX_QNN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// Initializes QNN runtime with specified backend:
// backend_type: 0 = Auto/HTP, 1 = HTP (Hexagon NPU), 2 = GPU, 3 = CPU
JAX_QNN_EXPORT int JAX_QNN_Init(int backend_type, const char* sdk_root);

// Returns active backend identifier string
JAX_QNN_EXPORT const char* JAX_QNN_GetBackendName();

// Compiles MLIR StableHLO text into an executable handle
JAX_QNN_EXPORT void* JAX_QNN_Compile(const char* mlir_text, size_t text_len);

// Frees the compiled executable handle
JAX_QNN_EXPORT void JAX_QNN_DestroyExecutable(void* exec_handle);

// Returns the number of inputs expected
JAX_QNN_EXPORT int JAX_QNN_GetNumInputs(void* exec_handle);

// Returns the number of outputs produced
JAX_QNN_EXPORT int JAX_QNN_GetNumOutputs(void* exec_handle);

// Returns output buffer byte size for a given output index
JAX_QNN_EXPORT int64_t JAX_QNN_GetOutputByteSize(void* exec_handle, int output_idx);

// Returns output rank (number of dimensions)
JAX_QNN_EXPORT int JAX_QNN_GetOutputRank(void* exec_handle, int output_idx);

// Returns output dimension sizes into dims_out (must have capacity >= rank)
JAX_QNN_EXPORT int JAX_QNN_GetOutputShape(void* exec_handle, int output_idx, int64_t* dims_out);

// Executes compiled graph with raw input and output memory buffers
JAX_QNN_EXPORT int JAX_QNN_Execute(
    void* exec_handle,
    const void* const* input_ptrs,
    const int64_t* input_bytes,
    int num_inputs,
    void* const* output_ptrs,
    const int64_t* output_bytes,
    int num_outputs
);

}

#endif  // JAX_QNN_C_API_H_
