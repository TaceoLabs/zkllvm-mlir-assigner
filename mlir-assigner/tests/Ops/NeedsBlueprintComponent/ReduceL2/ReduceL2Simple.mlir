module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "reducel2simple.mlir"} {
  func.func @main_graph(%arg0: memref<1x10xf32>) -> memref<1x1xf32> attributes {input_names = ["in_a"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x10xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        %0 = affine.load %arg0[%c0, %arg2] : memref<1x10xf32>
        %1 = affine.load %arg0[%c0, %arg2] : memref<1x10xf32>
        %2 = arith.mulf %0, %1 : f32
        affine.store %2, %alloc[%arg1, %arg2] : memref<1x10xf32>
      }
    }
    %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1x1xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.store %cst, %alloc_0[%arg1, %arg2] : memref<1x1xf32>
      }
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        %0 = affine.load %alloc[%arg1, %arg2] : memref<1x10xf32>
        %1 = affine.load %alloc_0[%arg1, %c0] : memref<1x1xf32>
        %2 = arith.addf %1, %0 : f32
        affine.store %2, %alloc_0[%arg1, %c0] : memref<1x1xf32>
      }
    }
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x1xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        %0 = affine.load %alloc_0[%arg1, %arg2] : memref<1x1xf32>
        %1 = math.sqrt %0 : f32
        affine.store %1, %alloc_1[%arg1, %arg2] : memref<1x1xf32>
      }
    }
    return %alloc_1 : memref<1x1xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22in_a\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
