module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "powpublicexponent.mlir"} {
  func.func @main_graph(%arg0: memref<1x10xf32>) -> memref<1x10xf32> attributes {input_names = ["in_a"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %c0 = arith.constant 0 : index
    %0 = "krnl.global"() {name = "constant_0", shape = [1, 10], value = dense<3.000000e+00> : tensor<1x10xf32>} : () -> memref<1x10xf32>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x10xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        %1 = affine.load %arg0[%c0, %arg2] : memref<1x10xf32>
        %2 = affine.load %0[%c0, %arg2] : memref<1x10xf32>
        %3 = math.powf %1, %2 : f32
        affine.store %3, %alloc[%arg1, %arg2] : memref<1x10xf32>
      }
    }
    return %alloc : memref<1x10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22in_a\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
