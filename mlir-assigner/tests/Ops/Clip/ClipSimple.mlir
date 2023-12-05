module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "clipsimple.0.mlir"} {
  func.func @main_graph(%arg0: memref<1x10xf32>) -> memref<1x10xf32> attributes {input_names = ["in_a"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %0 = "krnl.global"() {name = "constant_0", shape = [], value = dense<0.000000e+00> : tensor<f32>} : () -> memref<f32>
    %1 = "krnl.global"() {name = "constant_1", shape = [], value = dense<1.000000e+00> : tensor<f32>} : () -> memref<f32>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x10xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        %2 = affine.load %arg0[%arg1, %arg2] : memref<1x10xf32>
        %3 = affine.load %0[] : memref<f32>
        %4 = affine.load %1[] : memref<f32>
        %5 = arith.cmpf olt, %2, %3 : f32
        %6 = arith.select %5, %3, %2 : f32
        %7 = arith.cmpf olt, %6, %4 : f32
        %8 = arith.select %7, %6, %4 : f32
        affine.store %8, %alloc[%arg1, %arg2] : memref<1x10xf32>
      }
    }
    return %alloc : memref<1x10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22in_a\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
