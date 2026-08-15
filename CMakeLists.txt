# =============================================================================
# JAX-QNN: Qualcomm QNN Backend for JAX
# =============================================================================
#
# This CMakeLists.txt builds the PJRT plugin shared library (pjrt_qnn.dll/.so)
# that bridges JAX's PJRT interface to Qualcomm's QNN SDK.
#
# Prerequisites:
#   - CMake >= 3.20
#   - C++17 compiler (MSVC, GCC, or Clang)
#   - QNN SDK: set QNN_SDK_ROOT environment variable
#   - PJRT headers: automatically fetched from OpenXLA
#
# Build:
#   mkdir build && cd build
#   cmake .. -DQNN_SDK_ROOT=<path> -DCMAKE_BUILD_TYPE=Release
#   cmake --build . --config Release
#
# =============================================================================

cmake_minimum_required(VERSION 3.20)
project(jax_qnn VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# QNN SDK
# =============================================================================

if(NOT DEFINED QNN_SDK_ROOT AND DEFINED ENV{QNN_SDK_ROOT})
    set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
endif()

if(NOT DEFINED QNN_SDK_ROOT)
    message(WARNING
        "QNN_SDK_ROOT not set. Building in STUB mode (no actual QNN execution).\n"
        "Set -DQNN_SDK_ROOT=<path> or QNN_SDK_ROOT env var to enable QNN.")
    set(QNN_STUB_MODE ON)
else()
    set(QNN_STUB_MODE OFF)
    message(STATUS "QNN SDK Root: ${QNN_SDK_ROOT}")
    
    # Verify QNN SDK structure
    if(NOT EXISTS "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h")
        message(FATAL_ERROR
            "QNN SDK headers not found at ${QNN_SDK_ROOT}/include/QNN/\n"
            "Expected to find QnnInterface.h. Check your QNN_SDK_ROOT path.")
    endif()
endif()

# =============================================================================
# PJRT C API Headers (from OpenXLA)
# =============================================================================

# We vendor a minimal copy of the PJRT C API header.
# In production, you'd fetch from openxla/xla or use FetchContent.
set(PJRT_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pjrt")

if(NOT EXISTS "${PJRT_HEADERS_DIR}/pjrt_c_api.h")
    message(STATUS "PJRT headers not found, will use vendored copy from csrc/pjrt/")
    set(PJRT_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc/pjrt")
endif()

# =============================================================================
# Plugin Shared Library
# =============================================================================

set(PLUGIN_SOURCES
    csrc/pjrt_qnn_plugin.cc
    csrc/qnn_client.cc
    csrc/qnn_device.cc
    csrc/qnn_executable.cc
    csrc/qnn_buffer.cc
    csrc/stablehlo_to_qnn.cc
    csrc/qnn_runtime.cc
)

add_library(pjrt_qnn SHARED ${PLUGIN_SOURCES})

target_include_directories(pjrt_qnn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${PJRT_HEADERS_DIR}
)

if(QNN_STUB_MODE)
    target_compile_definitions(pjrt_qnn PRIVATE QNN_STUB_MODE=1)
    message(STATUS "Building in STUB mode — QNN calls will be simulated")
else()
    target_include_directories(pjrt_qnn PRIVATE
        ${QNN_SDK_ROOT}/include/QNN
        ${QNN_SDK_ROOT}/include
    )
    
    # QNN libraries are loaded dynamically at runtime via dlopen/LoadLibrary,
    # so we don't link against them at build time. This allows the plugin
    # to gracefully handle missing QNN libraries.
    target_compile_definitions(pjrt_qnn PRIVATE
        QNN_SDK_ROOT="${QNN_SDK_ROOT}"
    )
endif()

# Ensure GetPjRtApi is exported
if(WIN32)
    target_compile_definitions(pjrt_qnn PRIVATE JAX_QNN_BUILDING_DLL=1)
    set_target_properties(pjrt_qnn PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF
        PREFIX ""
        SUFFIX ".dll"
    )
else()
    set_target_properties(pjrt_qnn PROPERTIES
        PREFIX ""
        SUFFIX ".so"
    )
    target_link_options(pjrt_qnn PRIVATE -Wl,--no-undefined)
endif()

# On Linux, we need dl for dlopen
if(UNIX AND NOT APPLE)
    target_link_libraries(pjrt_qnn PRIVATE dl)
endif()

# =============================================================================
# Install
# =============================================================================

# Install the plugin DLL next to the Python package
install(TARGETS pjrt_qnn
    LIBRARY DESTINATION jax_plugins/qnn
    RUNTIME DESTINATION jax_plugins/qnn  # For .dll on Windows
)

message(STATUS "JAX-QNN plugin will be built as: $<TARGET_FILE:pjrt_qnn>")
