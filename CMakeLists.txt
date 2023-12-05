cmake_minimum_required(VERSION 3.19.0)
project(zkML-onnx-frontend)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(MLIR.cmake)
add_subdirectory(libs)
add_subdirectory(mlir-assigner)
add_subdirectory(zkml-onnx-compiler)
