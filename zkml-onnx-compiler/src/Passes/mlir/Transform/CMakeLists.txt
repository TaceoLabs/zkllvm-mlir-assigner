add_onnx_mlir_library(ZkMLIRTransform
 ElimCopySignPass.cpp

 LINK_LIBS PRIVATE

 OMSpecializedKernelOpInterface
 OMCompilerOptions
 OMONNXOps
 OMSupport
 MLIRTransforms
 MLIRAffineUtils
 )
