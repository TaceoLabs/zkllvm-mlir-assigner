module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "matmul1dim.0.mlir"} {
  func.func @main_graph(%arg0: memref<1x196xf32>, %arg1: memref<196x1xf32>) -> memref<1x1xf32> attributes {input_names = ["in_a", "in_b"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1xf32>
    %alloca = memref.alloca() : memref<f32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        affine.store %cst, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 196 {
          %1 = affine.load %arg0[%arg2, %arg4] : memref<1x196xf32>
          %2 = affine.load %arg1[%arg4, %arg3] : memref<196x1xf32>
          %3 = affine.load %alloca[] : memref<f32>
          %4 = arith.mulf %1, %2 : f32
          %5 = arith.addf %3, %4 : f32
          affine.store %5, %alloca[] : memref<f32>
        }
        %0 = affine.load %alloca[] : memref<f32>
        affine.store %0, %alloc[%arg2, %arg3] : memref<1x1xf32>
      }
    }
    return %alloc : memref<1x1xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 196] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [196 , 1] , \22name\22 : \22in_b\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
