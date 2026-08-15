// =============================================================================
// pjrt_qnn_plugin.h — Main entry point header for QNN PJRT plugin
// =============================================================================

#ifndef JAX_QNN_PJRT_QNN_PLUGIN_H_
#define JAX_QNN_PJRT_QNN_PLUGIN_H_

#include "pjrt/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Main exported symbol called by JAX when loading the plugin library.
PJRT_PLUGIN_EXPORT const PJRT_Api* GetPjRtApi();

#ifdef __cplusplus
}
#endif

#endif  // JAX_QNN_PJRT_QNN_PLUGIN_H_
