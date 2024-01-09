module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "batchnormalizationsimple.mlir"} {
  func.func @main_graph(%arg0: memref<1x8x32x32xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) -> memref<1x8x32x32xf32> attributes {input_names = ["in_a", "in_d", "in_e"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %c0 = arith.constant 0 : index
    %0 = "krnl.global"() {name = "constant_1", shape = [1], value = dense<9.99999974E-6> : tensor<1xf32>} : () -> memref<1xf32>
    %1 = "krnl.global"() {name = "constant_2", shape = [8], value = dense<[1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<8xf32>} : () -> memref<8xf32>
    %2 = "krnl.global"() {name = "constant_3", shape = [8], value = dense<[0.000000e+00, 0.000000e+00, 1.000000e-01, 1.000000e-01, 2.000000e-01, 2.000000e-01, 3.000000e-01, 3.000000e-01]> : tensor<8xf32>} : () -> memref<8xf32>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<8xf32>
    affine.for %arg3 = 0 to 8 {
      %3 = affine.load %arg2[%arg3] : memref<8xf32>
      %4 = affine.load %0[%c0] : memref<1xf32>
      %5 = arith.addf %3, %4 : f32
      %6 = math.sqrt %5 : f32
      %7 = affine.load %1[%arg3] : memref<8xf32>
      %8 = arith.divf %7, %6 : f32
      affine.store %8, %alloc[%arg3] : memref<8xf32>
    }
    %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [8, 1, 1], strides: [1, 1, 1] : memref<8xf32> to memref<8x1x1xf32>
    %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1x8x32x32xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 32 {
          affine.for %arg6 = 0 to 32 {
            %3 = affine.load %arg0[%c0, %arg4, %arg5, %arg6] : memref<1x8x32x32xf32>
            %4 = affine.load %reinterpret_cast[%arg4, %c0, %c0] : memref<8x1x1xf32>
            %5 = arith.mulf %3, %4 : f32
            affine.store %5, %alloc_0[%arg3, %arg4, %arg5, %arg6] : memref<1x8x32x32xf32>
          }
        }
      }
    }
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<8xf32>
    affine.for %arg3 = 0 to 8 {
      %3 = affine.load %arg1[%arg3] : memref<8xf32>
      %4 = affine.load %alloc[%arg3] : memref<8xf32>
      %5 = arith.mulf %3, %4 : f32
      %6 = affine.load %2[%arg3] : memref<8xf32>
      %7 = arith.subf %6, %5 : f32
      affine.store %7, %alloc_1[%arg3] : memref<8xf32>
    }
    %reinterpret_cast_2 = memref.reinterpret_cast %alloc_1 to offset: [0], sizes: [8, 1, 1], strides: [1, 1, 1] : memref<8xf32> to memref<8x1x1xf32>
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1x8x32x32xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 32 {
          affine.for %arg6 = 0 to 32 {
            %3 = affine.load %alloc_0[%c0, %arg4, %arg5, %arg6] : memref<1x8x32x32xf32>
            %4 = affine.load %reinterpret_cast_2[%arg4, %c0, %c0] : memref<8x1x1xf32>
            %5 = arith.addf %3, %4 : f32
            affine.store %5, %alloc_3[%arg3, %arg4, %arg5, %arg6] : memref<1x8x32x32xf32>
          }
        }
      }
    }
    return %alloc_3 : memref<1x8x32x32xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 8 , 32 , 32] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [8] , \22name\22 : \22in_d\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [8] , \22name\22 : \22in_e\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 8 , 32 , 32] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
