module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "gemmalphabetaboth.0.mlir"} {
  func.func @main_graph(%arg0: memref<1x196xf32>, %arg1: memref<196x128xf32>, %arg2: memref<128xf32>) -> memref<1x128xf32> attributes {input_names = ["in_a", "in_b", "in_c"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.020000e+01 : f32
    %cst_1 = arith.constant -3.450000e+00 : f32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<1x128xf32>
    %alloca = memref.alloca() : memref<f32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.store %cst, %alloca[] : memref<f32>
        affine.for %arg5 = 0 to 196 {
          %5 = affine.load %arg0[%arg3, %arg5] : memref<1x196xf32>
          %6 = affine.load %arg1[%arg5, %arg4] : memref<196x128xf32>
          %7 = arith.mulf %5, %6 : f32
          %8 = affine.load %alloca[] : memref<f32>
          %9 = arith.addf %7, %8 : f32
          affine.store %9, %alloca[] : memref<f32>
        }
        %0 = affine.load %alloca[] : memref<f32>
        %1 = arith.mulf %0, %cst_1 : f32
        %2 = affine.load %arg2[%arg4] : memref<128xf32>
        %3 = arith.mulf %2, %cst_0 : f32
        %4 = arith.addf %1, %3 : f32
        affine.store %4, %alloc[%arg3, %arg4] : memref<1x128xf32>
      }
    }
    return %alloc : memref<1x128xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 196] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [196 , 128] , \22name\22 : \22in_b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [128] , \22name\22 : \22in_c\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 128] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
