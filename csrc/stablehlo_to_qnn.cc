// =============================================================================
// stablehlo_to_qnn.cc — StableHLO/MLIR parser for QNN backend
// =============================================================================
//
// This is a text-based parser for the MLIR output that JAX produces.
// When JAX compiles a function with jax.jit(backend="qnn"), it produces
// a StableHLO MLIR module like:
//
//   module @jit_f attributes {mhlo.num_replicas = 1 : i32} {
//     func.func public @main(%arg0: tensor<1024x1024xf32>,
//                            %arg1: tensor<1024x1024xf32>)
//         -> tensor<1024x1024xf32> {
//       %0 = stablehlo.add %arg0, %arg1 : tensor<1024x1024xf32>
//       return %0 : tensor<1024x1024xf32>
//     }
//   }
//
// We parse this text to extract:
// 1. Input shapes/types from the function signature
// 2. Operations (op type, operands, result types)
// 3. Output shapes/types from the return statement
//
// NOTE: This is a simplified regex-free parser. For production, you would
// use MLIR's C API or link against the MLIR parser library. This approach
// is chosen to minimize build dependencies for the MVP.
//
// =============================================================================

#include "stablehlo_to_qnn.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace jax_qnn {

// =============================================================================
// Helper: parse a tensor type like "tensor<1024x1024xf32>"
// =============================================================================

struct TensorType {
  std::vector<int64_t> shape;
  PJRT_Buffer_Type element_type = PJRT_Buffer_Type_F32;
  bool valid = false;
};

static PJRT_Buffer_Type ParseElementType(const std::string& type_str) {
  if (type_str == "f32") return PJRT_Buffer_Type_F32;
  if (type_str == "f64") return PJRT_Buffer_Type_F64;
  if (type_str == "f16") return PJRT_Buffer_Type_F16;
  if (type_str == "bf16") return PJRT_Buffer_Type_BF16;
  if (type_str == "i8" || type_str == "si8") return PJRT_Buffer_Type_S8;
  if (type_str == "i16" || type_str == "si16") return PJRT_Buffer_Type_S16;
  if (type_str == "i32" || type_str == "si32") return PJRT_Buffer_Type_S32;
  if (type_str == "i64" || type_str == "si64") return PJRT_Buffer_Type_S64;
  if (type_str == "ui8") return PJRT_Buffer_Type_U8;
  if (type_str == "ui16") return PJRT_Buffer_Type_U16;
  if (type_str == "ui32") return PJRT_Buffer_Type_U32;
  if (type_str == "ui64") return PJRT_Buffer_Type_U64;
  if (type_str == "i1" || type_str == "pred") return PJRT_Buffer_Type_PRED;
  return PJRT_Buffer_Type_F32;  // default
}

static TensorType ParseTensorType(const std::string& str) {
  TensorType result;
  
  // Find "tensor<...>"
  auto start = str.find("tensor<");
  if (start == std::string::npos) return result;
  start += 7;  // skip "tensor<"
  
  auto end = str.find('>', start);
  if (end == std::string::npos) return result;
  
  std::string inner = str.substr(start, end - start);
  
  // Handle scalar tensor "tensor<f32>"
  if (inner.find('x') == std::string::npos) {
    result.element_type = ParseElementType(inner);
    result.valid = true;
    return result;
  }
  
  // Parse dimensions: "1024x1024xf32"
  std::istringstream iss(inner);
  std::string token;
  std::vector<std::string> parts;
  while (std::getline(iss, token, 'x')) {
    parts.push_back(token);
  }
  
  if (parts.empty()) return result;
  
  // Last part is the element type
  result.element_type = ParseElementType(parts.back());
  
  // All other parts are dimensions
  for (size_t i = 0; i < parts.size() - 1; ++i) {
    try {
      // Handle dynamic dimensions marked with '?'
      if (parts[i] == "?") {
        result.shape.push_back(-1);  // Dynamic
      } else {
        result.shape.push_back(std::stoll(parts[i]));
      }
    } catch (...) {
      return result;
    }
  }
  
  result.valid = true;
  return result;
}

// =============================================================================
// Helper: trim whitespace
// =============================================================================

static std::string Trim(const std::string& s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

// =============================================================================
// Helper: parse a value reference like "%0", "%arg0", "%1"
// =============================================================================

static int ParseValueRef(const std::string& ref, 
                         const std::unordered_map<std::string, int>& value_map) {
  auto trimmed = Trim(ref);
  // Remove trailing type annotation if present (e.g., "%0 : tensor<...>")
  auto colon = trimmed.find(':');
  if (colon != std::string::npos) {
    trimmed = Trim(trimmed.substr(0, colon));
  }
  // Remove comma
  while (!trimmed.empty() && (trimmed.back() == ',' || trimmed.back() == ')')) {
    trimmed.pop_back();
  }
  trimmed = Trim(trimmed);
  
  auto it = value_map.find(trimmed);
  if (it != value_map.end()) {
    return it->second;
  }
  return -1;
}

// =============================================================================
// Main parser
// =============================================================================

std::unique_ptr<ParsedComputation> ParseStableHLO(const std::string& mlir_text) {
  auto comp = std::make_unique<ParsedComputation>();
  comp->raw_mlir = mlir_text;
  
  std::cerr << "[JAX-QNN] Parsing StableHLO (" << mlir_text.size() 
            << " bytes)" << std::endl;
  
  // Value name → index mapping
  std::unordered_map<std::string, int> value_map;
  int next_value_idx = 0;
  
  // Parse line by line
  std::istringstream stream(mlir_text);
  std::string line;
  bool in_function = false;
  
  while (std::getline(stream, line)) {
    std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '/') continue;  // skip comments
    
    // Parse module name
    if (trimmed.find("module") != std::string::npos) {
      auto at = trimmed.find("@");
      if (at != std::string::npos) {
        auto end = trimmed.find_first_of(" {(", at);
        comp->name = trimmed.substr(at + 1, end - at - 1);
      }
      continue;
    }
    
    // Parse function signature
    if (trimmed.find("func.func") != std::string::npos || 
        trimmed.find("func @") != std::string::npos) {
      in_function = true;
      
      // Extract function name
      auto at = trimmed.find("@");
      if (at != std::string::npos) {
        auto end = trimmed.find_first_of("(", at);
        if (end != std::string::npos) {
          comp->name = trimmed.substr(at + 1, end - at - 1);
        }
      }
      
      // Parse arguments: find all %argN: tensor<...> patterns
      // We need to handle multi-line signatures
      std::string full_sig = trimmed;
      while (full_sig.find('{') == std::string::npos && std::getline(stream, line)) {
        full_sig += " " + Trim(line);
      }
      
      // Find argument types
      size_t pos = 0;
      while ((pos = full_sig.find("%arg", pos)) != std::string::npos) {
        // Extract arg name
        auto name_end = full_sig.find_first_of(":,)", pos);
        std::string arg_name = Trim(full_sig.substr(pos, name_end - pos));
        
        // Find its type
        auto type_start = full_sig.find("tensor<", pos);
        if (type_start != std::string::npos && type_start < full_sig.find("%arg", pos + 1)) {
          auto type = ParseTensorType(full_sig.substr(type_start));
          if (type.valid) {
            value_map[arg_name] = next_value_idx++;
            comp->input_shapes.push_back(type.shape);
            comp->input_types.push_back(type.element_type);
          }
        }
        pos = name_end + 1;
      }
      
      comp->num_inputs = comp->input_shapes.size();
      
      // Parse return type from "-> tensor<...>" or "-> (tensor<...>, tensor<...>)"
      auto arrow = full_sig.find("->");
      if (arrow != std::string::npos) {
        std::string ret_types = full_sig.substr(arrow + 2);
        size_t rpos = 0;
        while ((rpos = ret_types.find("tensor<", rpos)) != std::string::npos) {
          auto type = ParseTensorType(ret_types.substr(rpos));
          if (type.valid) {
            comp->output_shapes.push_back(type.shape);
            comp->output_types.push_back(type.element_type);
          }
          rpos += 7;  // advance past "tensor<"
        }
        comp->num_outputs = comp->output_shapes.size();
      }
      
      continue;
    }
    
    if (!in_function) continue;
    
    // Parse return statement
    if (trimmed.find("return ") == 0 || trimmed.find("stablehlo.return ") == 0) {
      ParsedOp op;
      op.op_type = "stablehlo.return";
      
      // Parse return operands
      auto space = trimmed.find(' ');
      if (space != std::string::npos) {
        std::string operands = trimmed.substr(space + 1);
        // Extract %references before ':'
        auto colon = operands.find(':');
        if (colon != std::string::npos) {
          operands = operands.substr(0, colon);
        }
        std::istringstream ops_stream(operands);
        std::string ref;
        while (std::getline(ops_stream, ref, ',')) {
          int idx = ParseValueRef(ref, value_map);
          if (idx >= 0) op.input_indices.push_back(idx);
        }
      }
      
      comp->ops.push_back(op);
      continue;
    }
    
    // Parse closing brace
    if (trimmed == "}" || trimmed == "})") {
      in_function = false;
      continue;
    }
    
    // Parse operation: %result = stablehlo.op %operand1, %operand2 : type
    if (trimmed.find('%') == 0 && trimmed.find('=') != std::string::npos) {
      ParsedOp op;
      
      // Extract result name
      auto eq = trimmed.find('=');
      std::string result_name = Trim(trimmed.substr(0, eq));
      
      // Handle multiple results: %0, %1 = ...
      // For now, assume single result
      int result_idx = next_value_idx++;
      value_map[result_name] = result_idx;
      op.output_indices.push_back(result_idx);
      
      // Extract op type and operands
      std::string rhs = Trim(trimmed.substr(eq + 1));
      
      // Find the op name (first word, may include namespace like "stablehlo.")
      auto first_space = rhs.find_first_of(" (");
      if (first_space != std::string::npos) {
        op.op_type = Trim(rhs.substr(0, first_space));
        
        // Parse operands
        std::string rest = rhs.substr(first_space);
        auto colon = rest.find(':');
        std::string operand_str = (colon != std::string::npos) ? 
            rest.substr(0, colon) : rest;
        
        // Extract %references
        size_t opos = 0;
        while ((opos = operand_str.find('%', opos)) != std::string::npos) {
          auto oend = operand_str.find_first_of(",: )\n{", opos);
          if (oend == std::string::npos) oend = operand_str.size();
          std::string ref = Trim(operand_str.substr(opos, oend - opos));
          int idx = ParseValueRef(ref, value_map);
          if (idx >= 0) op.input_indices.push_back(idx);
          opos = oend + 1;
        }
        
        // Parse result type
        if (colon != std::string::npos) {
          std::string type_str = rest.substr(colon + 1);
          auto type = ParseTensorType(type_str);
          if (type.valid) {
            op.output_shapes.push_back(type.shape);
            op.output_types.push_back(type.element_type);
          }
        }
      } else {
        op.op_type = rhs;
      }
      
      // Handle special op patterns
      // "stablehlo.constant" — may have dense attribute
      if (op.op_type == "stablehlo.constant") {
        // TODO: parse dense<...> values
      }
      
      comp->ops.push_back(op);
      continue;
    }
  }
  
  // If no explicit outputs were parsed from return type, infer from return op
  if (comp->num_outputs == 0 && !comp->ops.empty()) {
    // Find the return op
    for (const auto& op : comp->ops) {
      if (op.op_type == "stablehlo.return") {
        comp->num_outputs = op.input_indices.size();
        // Try to get output shapes from the values
        break;
      }
    }
    if (comp->num_outputs == 0) {
      comp->num_outputs = 1;  // Assume single output
    }
  }
  
  // If output shapes are still empty, try to infer from the last non-return op
  if (comp->output_shapes.empty() && !comp->ops.empty()) {
    for (int i = comp->ops.size() - 1; i >= 0; --i) {
      if (comp->ops[i].op_type != "stablehlo.return" && 
          !comp->ops[i].output_shapes.empty()) {
        comp->output_shapes = comp->ops[i].output_shapes;
        comp->output_types = comp->ops[i].output_types;
        break;
      }
    }
  }
  
  std::cerr << "[JAX-QNN] Parsed computation '" << comp->name << "': "
            << comp->num_inputs << " inputs, " << comp->num_outputs 
            << " outputs, " << comp->ops.size() << " ops" << std::endl;
  
  for (size_t i = 0; i < comp->input_shapes.size(); ++i) {
    std::cerr << "[JAX-QNN]   Input " << i << ": [";
    for (size_t j = 0; j < comp->input_shapes[i].size(); ++j) {
      if (j > 0) std::cerr << ", ";
      std::cerr << comp->input_shapes[i][j];
    }
    std::cerr << "] " << PjrtTypeName(comp->input_types[i]) << std::endl;
  }
  
  for (const auto& op : comp->ops) {
    std::cerr << "[JAX-QNN]   Op: " << op.op_type 
              << " (inputs: " << op.input_indices.size()
              << ", outputs: " << op.output_shapes.size() << ")" << std::endl;
  }
  
  return comp;
}

}  // namespace jax_qnn
