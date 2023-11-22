module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "softmaxbasicmnist.0.mlir"} {
  func.func @main_graph(%arg0: memref<1x10xf32>) -> memref<1x10xf32> attributes {input_names = ["in_a"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x10xf32>
    %alloc_1 = memref.alloc() : memref<f32>
    %alloc_2 = memref.alloc() : memref<f32>
    affine.for %arg1 = 0 to 1 {
      affine.store %cst_0, %alloc_1[] : memref<f32>
      affine.store %cst, %alloc_2[] : memref<f32>
      affine.for %arg2 = 0 to 10 {
        %2 = affine.load %alloc_2[] : memref<f32>
        %3 = affine.load %arg0[%arg1, %arg2] : memref<1x10xf32>
        %4 = arith.cmpf ogt, %2, %3 : f32
        %5 = arith.select %4, %2, %3 : f32
        affine.store %5, %alloc_2[] : memref<f32>
      }
      %0 = affine.load %alloc_2[] : memref<f32>
      affine.for %arg2 = 0 to 10 {
        %2 = affine.load %alloc_1[] : memref<f32>
        %3 = affine.load %arg0[%arg1, %arg2] : memref<1x10xf32>
        %4 = arith.subf %3, %0 : f32
        %5 = math.exp %4 : f32
        %6 = arith.addf %2, %5 : f32
        affine.store %6, %alloc_1[] : memref<f32>
        affine.store %5, %alloc[%arg1, %arg2] : memref<1x10xf32>
      }
      %1 = affine.load %alloc_1[] : memref<f32>
      affine.for %arg2 = 0 to 10 {
        %2 = affine.load %alloc[%arg1, %arg2] : memref<1x10xf32>
        %3 = arith.divf %2, %1 : f32
        affine.store %3, %alloc[%arg1, %arg2] : memref<1x10xf32>
      }
    }
    return %alloc : memref<1x10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22in_a\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
