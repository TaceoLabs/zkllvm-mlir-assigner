module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "rnnsimple.mlir"} {
  func.func @main_graph(%arg0: memref<1x1x10xf32>, %arg1: memref<1x2x10xf32>, %arg2: memref<1x2x2xf32>) -> (memref<1x1x1x2xf32>, memref<1x1x2xf32>) attributes {input_names = ["in_a", "in_b", "in_c"], llvm.emit_c_interface, output_names = ["out_a", "out_b"]} {
    %c2_i64 = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x2xf32>
    %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1x1x2xf32>
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x2xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 2 {
        affine.store %cst, %alloc_1[%arg3, %arg4] : memref<1x2xf32>
      }
    }
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [2, 10], strides: [10, 1] : memref<1x2x10xf32> to memref<2x10xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2, 2], strides: [2, 1] : memref<1x2x2xf32> to memref<2x2xf32>
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<10x2xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 10 {
        %0 = affine.load %reinterpret_cast[%arg3, %arg4] : memref<2x10xf32>
        affine.store %0, %alloc_3[%arg4, %arg3] : memref<10x2xf32>
      }
    }
    %alloc_4 = memref.alloc() {alignment = 16 : i64} : memref<2x2xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.load %reinterpret_cast_2[%arg3, %arg4] : memref<2x2xf32>
        affine.store %0, %alloc_4[%arg4, %arg3] : memref<2x2xf32>
      }
    }
    affine.for %arg3 = 0 to 1 {
      %alloc_5 = memref.alloc() {alignment = 16 : i64} : memref<1x10xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 10 {
          %0 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<1x1x10xf32>
          affine.store %0, %alloc_5[%arg4, %arg5] : memref<1x10xf32>
        }
      }
      %alloc_6 = memref.alloc() {alignment = 16 : i64} : memref<1x2xf32>
      %alloc_7 = memref.alloc() : memref<10xf32>
      %alloc_8 = memref.alloc() : memref<10xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 2 {
          affine.for %arg6 = 0 to 10 {
            %1 = affine.load %alloc_5[%arg4, %arg6] : memref<1x10xf32>
            %2 = affine.load %alloc_3[%arg6, %arg5] : memref<10x2xf32>
            affine.store %1, %alloc_7[%arg6] : memref<10xf32>
            affine.store %2, %alloc_8[%arg6] : memref<10xf32>
          }
          %0 = "zkML.dot-product"(%alloc_7, %alloc_8) : (memref<10xf32>, memref<10xf32>) -> f32
          affine.store %0, %alloc_6[%arg4, %arg5] : memref<1x2xf32>
        }
      }
      memref.dealloc %alloc_7 : memref<10xf32>
      memref.dealloc %alloc_8 : memref<10xf32>
      %alloc_9 = memref.alloc() {alignment = 16 : i64} : memref<1x2xf32>
      %alloc_10 = memref.alloc() : memref<2xf32>
      %alloc_11 = memref.alloc() : memref<2xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 2 {
          affine.for %arg6 = 0 to 2 {
            %1 = affine.load %alloc_1[%arg4, %arg6] : memref<1x2xf32>
            %2 = affine.load %alloc_4[%arg6, %arg5] : memref<2x2xf32>
            affine.store %1, %alloc_10[%arg6] : memref<2xf32>
            affine.store %2, %alloc_11[%arg6] : memref<2xf32>
          }
          %0 = "zkML.dot-product"(%alloc_10, %alloc_11) : (memref<2xf32>, memref<2xf32>) -> f32
          affine.store %0, %alloc_9[%arg4, %arg5] : memref<1x2xf32>
        }
      }
      memref.dealloc %alloc_10 : memref<2xf32>
      memref.dealloc %alloc_11 : memref<2xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 2 {
          %0 = affine.load %alloc_6[%arg4, %arg5] : memref<1x2xf32>
          %1 = affine.load %alloc_9[%arg4, %arg5] : memref<1x2xf32>
          %2 = arith.addf %0, %1 : f32
          %3 = math.tanh %2 : f32
          affine.store %3, %alloc_1[%arg4, %arg5] : memref<1x2xf32>
          affine.store %3, %alloc[%arg3, %c0, %arg4, %arg5] : memref<1x1x1x2xf32>
        }
      }
    }
    "krnl.memcpy"(%alloc_0, %alloc_1, %c2_i64, %c0, %c0) : (memref<1x1x2xf32>, memref<1x2xf32>, i64, index, index) -> ()
    return %alloc, %alloc_0 : memref<1x1x1x2xf32>, memref<1x1x2xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 10] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 2 , 10] , \22name\22 : \22in_b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 2 , 2] , \22name\22 : \22in_c\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 1 , 2] , \22name\22 : \22out_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 2] , \22name\22 : \22out_b\22 }\0A\0A]\00"} : () -> ()
}
