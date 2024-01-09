module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "einsuminnerprod.mlir"} {
  func.func @main_graph(%arg0: memref<5xf32>, %arg1: memref<5xf32>) -> memref<f32> attributes {input_names = ["in_a", "in_b"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 5], strides: [5, 1] : memref<5xf32> to memref<1x5xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [5, 1], strides: [1, 1] : memref<5xf32> to memref<5x1xf32>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1xf32>
    %alloc_1 = memref.alloc() : memref<5xf32>
    %alloc_2 = memref.alloc() : memref<5xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        affine.for %arg4 = 0 to 5 {
          %1 = affine.load %reinterpret_cast[%arg2, %arg4] : memref<1x5xf32>
          %2 = affine.load %reinterpret_cast_0[%arg4, %arg3] : memref<5x1xf32>
          affine.store %1, %alloc_1[%arg4] : memref<5xf32>
          affine.store %2, %alloc_2[%arg4] : memref<5xf32>
        }
        %0 = "zkML.dot-product"(%alloc_1, %alloc_2) : (memref<5xf32>, memref<5xf32>) -> f32
        affine.store %0, %alloc[%arg2, %arg3] : memref<1x1xf32>
      }
    }
    memref.dealloc %alloc_1 : memref<5xf32>
    memref.dealloc %alloc_2 : memref<5xf32>
    %reinterpret_cast_3 = memref.reinterpret_cast %alloc to offset: [0], sizes: [], strides: [] : memref<1x1xf32> to memref<f32>
    return %reinterpret_cast_3 : memref<f32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [5] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [5] , \22name\22 : \22in_b\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
