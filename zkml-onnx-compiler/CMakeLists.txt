cmake_minimum_required(VERSION 3.19.0)

add_onnx_mlir_library(ZkMLIRTransform
 src/Passes/mlir/Transform/PowFToGenericExpPass.cpp
 src/Passes/mlir/Transform/ElimCopySignPass.cpp
 src/Passes/mlir/Transform/AddTracingOperationsPass.cpp
 src/Passes/mlir/Transform/SetPrivateInputPass.cpp

 LINK_LIBS PRIVATE

 MLIRTransforms
 MLIRFuncDialect
 MLIRMathDialect

 OMONNXOps
 OMKrnlOps
 MLIRZkMlDialect
 )

add_onnx_mlir_executable(zkml-onnx-compiler
  src/zkml-onnx-compiler.cpp

  LINK_LIBS PRIVATE

  OMCompilerOptions
  OMCompilerUtils
  ZkMLIRTransform
  
  MLIRZkMlDialect
  MLIROpenMPToLLVMIRTranslation
  )

target_include_directories(ZkMLIRTransform PUBLIC include)
target_include_directories(zkml-onnx-compiler PUBLIC include)
