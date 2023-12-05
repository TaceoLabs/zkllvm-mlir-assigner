add_onnx_mlir_library(ZkMLIRConversion
  AffineFullUnrollPass.cpp
  AffineFullUnrollPattern.cpp
  #RewriteMulOps.cpp

  LINK_LIBS PRIVATE

  OMSpecializedKernelOpInterface
  OMCompilerOptions
  OMONNXOps
  OMSupport
  MLIRTransforms
  MLIRAffineUtils
  OMMlirDialects
  )


