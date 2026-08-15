// =============================================================================
// stablehlo_to_qnn.h — StableHLO/MLIR parser and QNN lowering
// =============================================================================

#ifndef JAX_QNN_STABLEHLO_TO_QNN_H_
#define JAX_QNN_STABLEHLO_TO_QNN_H_

#include "qnn_types.h"
#include <memory>
#include <string>

namespace jax_qnn {

// Parse a serialized StableHLO/MLIR module into our internal representation.
//
// JAX sends programs in one of these formats:
// - "mlir" format: text MLIR with StableHLO dialect ops
// - "hlo" format: serialized HLO proto (we don't support this yet)
//
// The parser extracts:
// - Function signature (input/output shapes and types)
// - Operation sequence
// - Constants
//
// Returns nullptr on parse failure.
std::unique_ptr<ParsedComputation> ParseStableHLO(const std::string& mlir_text);

}  // namespace jax_qnn

#endif  // JAX_QNN_STABLEHLO_TO_QNN_H_
