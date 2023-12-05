# Must unset LLVM_DIR in cache. Otherwise, when MLIR_DIR changes LLVM_DIR
# won't change accordingly.
unset(LLVM_DIR CACHE)
if (NOT DEFINED MLIR_DIR)
  message(FATAL_ERROR "MLIR_DIR is not configured but it is required. "
    "Set the cmake option MLIR_DIR, e.g.,\n"
    "    cmake -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir ..\n"
    )
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})
