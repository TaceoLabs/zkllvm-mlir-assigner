module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "topksimple.mlir"} {
  func.func @main_graph(%arg0: memref<1x10xf32>) -> (memref<1x3xf32>, memref<1x3xi64>) attributes {input_names = ["in_a"], llvm.emit_c_interface, output_names = ["out_a", "out_b"]} {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x3xf32>
    %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1x3xi64>
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x10xindex>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        affine.store %arg2, %alloc_1[%arg1, %arg2] : memref<1x10xindex>
      }
    }
    "krnl.call"(%alloc_1, %arg0, %c1_i64, %c0_i64) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<1x10xindex>, memref<1x10xf32>, i64, i64) -> ()
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 3 {
        %0 = affine.load %alloc_1[%arg1, %arg2] : memref<1x10xindex>
        %1 = memref.load %arg0[%arg1, %0] : memref<1x10xf32>
        affine.store %1, %alloc[%arg1, %arg2] : memref<1x3xf32>
        %2 = arith.index_cast %0 : index to i64
        affine.store %2, %alloc_0[%arg1, %arg2] : memref<1x3xi64>
      }
    }
    return %alloc, %alloc_0 : memref<1x3xf32>, memref<1x3xi64>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22in_a\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3] , \22name\22 : \22out_a\22 }\0A ,    { \22type\22 : \22i64\22 , \22dims\22 : [1 , 3] , \22name\22 : \22out_b\22 }\0A\0A]\00"} : () -> ()
}
