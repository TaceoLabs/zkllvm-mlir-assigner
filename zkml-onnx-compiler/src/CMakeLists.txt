add_subdirectory(Passes)

add_onnx_mlir_executable(zkml-onnx-compiler
  zkml-onnx-compiler.cpp

  LINK_LIBS PRIVATE

  OMCompilerOptions
  OMCompilerUtils
  ZkMLIRAnalysis
  ZkMLIRConversion 
  ZkMLIRTransform
  
  MLIRZkMlDialect
  )
