#map = affine_map<()[s0] -> (s0 + 4)>
#map1 = affine_map<()[s0] -> (s0 + 6)>
#map2 = affine_map<()[s0] -> (s0 + 2)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "lstmsimple.mlir"} {
  func.func @main_graph(%arg0: memref<1x1x10xf32>, %arg1: memref<1x8x10xf32>, %arg2: memref<1x8x2xf32>) -> (memref<1x1x1x2xf32>, memref<1x1x2xf32>, memref<1x1x2xf32>) attributes {input_names = ["in_a", "in_b", "in_c"], llvm.emit_c_interface, output_names = ["out_a", "out_b", "out_c"]} {
    %c2_i64 = arith.constant 2 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x2xf32>
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x1x2xf32>
    %alloc_2 = memref.alloc() {alignment = 16 : i64} : memref<1x1x2xf32>
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1x2xf32>
    %alloc_4 = memref.alloc() {alignment = 16 : i64} : memref<1x2xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 2 {
        affine.store %cst_0, %alloc_3[%arg3, %arg4] : memref<1x2xf32>
        affine.store %cst_0, %alloc_4[%arg3, %arg4] : memref<1x2xf32>
      }
    }
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [8, 10], strides: [10, 1] : memref<1x8x10xf32> to memref<8x10xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [8, 2], strides: [2, 1] : memref<1x8x2xf32> to memref<8x2xf32>
    %alloc_6 = memref.alloc() {alignment = 16 : i64} : memref<10x8xf32>
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 10 {
        %0 = affine.load %reinterpret_cast[%arg3, %arg4] : memref<8x10xf32>
        affine.store %0, %alloc_6[%arg4, %arg3] : memref<10x8xf32>
      }
    }
    %alloc_7 = memref.alloc() {alignment = 16 : i64} : memref<2x8xf32>
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.load %reinterpret_cast_5[%arg3, %arg4] : memref<8x2xf32>
        affine.store %0, %alloc_7[%arg4, %arg3] : memref<2x8xf32>
      }
    }
    affine.for %arg3 = 0 to 1 {
      %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<1x10xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 10 {
          %0 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<1x1x10xf32>
          affine.store %0, %alloc_8[%arg4, %arg5] : memref<1x10xf32>
        }
      }
      %alloc_9 = memref.alloc() {alignment = 16 : i64} : memref<1x8xf32>
      %alloc_10 = memref.alloc() : memref<10xf32>
      %alloc_11 = memref.alloc() : memref<10xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 10 {
            %1 = affine.load %alloc_8[%arg4, %arg6] : memref<1x10xf32>
            %2 = affine.load %alloc_6[%arg6, %arg5] : memref<10x8xf32>
            affine.store %1, %alloc_10[%arg6] : memref<10xf32>
            affine.store %2, %alloc_11[%arg6] : memref<10xf32>
          }
          %0 = "zkML.dot-product"(%alloc_10, %alloc_11) : (memref<10xf32>, memref<10xf32>) -> f32
          affine.store %0, %alloc_9[%arg4, %arg5] : memref<1x8xf32>
        }
      }
      memref.dealloc %alloc_10 : memref<10xf32>
      memref.dealloc %alloc_11 : memref<10xf32>
      %alloc_12 = memref.alloc() {alignment = 16 : i64} : memref<1x8xf32>
      %alloc_13 = memref.alloc() : memref<2xf32>
      %alloc_14 = memref.alloc() : memref<2xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 2 {
            %1 = affine.load %alloc_3[%arg4, %arg6] : memref<1x2xf32>
            %2 = affine.load %alloc_7[%arg6, %arg5] : memref<2x8xf32>
            affine.store %1, %alloc_13[%arg6] : memref<2xf32>
            affine.store %2, %alloc_14[%arg6] : memref<2xf32>
          }
          %0 = "zkML.dot-product"(%alloc_13, %alloc_14) : (memref<2xf32>, memref<2xf32>) -> f32
          affine.store %0, %alloc_12[%arg4, %arg5] : memref<1x8xf32>
        }
      }
      memref.dealloc %alloc_13 : memref<2xf32>
      memref.dealloc %alloc_14 : memref<2xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 2 {
          %0 = affine.load %alloc_4[%arg4, %arg5] : memref<1x2xf32>
          %1 = affine.load %alloc_9[%arg4, %arg5] : memref<1x8xf32>
          %2 = affine.load %alloc_12[%arg4, %arg5] : memref<1x8xf32>
          %3 = arith.addf %1, %2 : f32
          %4 = arith.subf %cst_0, %3 : f32
          %5 = math.exp %4 : f32
          %6 = arith.addf %5, %cst : f32
          %7 = arith.divf %cst, %6 : f32
          %8 = affine.apply #map()[%arg5]
          %9 = affine.load %alloc_9[%arg4, %8] : memref<1x8xf32>
          %10 = affine.apply #map()[%arg5]
          %11 = affine.load %alloc_12[%arg4, %10] : memref<1x8xf32>
          %12 = arith.addf %9, %11 : f32
          %13 = arith.subf %cst_0, %12 : f32
          %14 = math.exp %13 : f32
          %15 = arith.addf %14, %cst : f32
          %16 = arith.divf %cst, %15 : f32
          %17 = affine.apply #map1()[%arg5]
          %18 = affine.load %alloc_9[%arg4, %17] : memref<1x8xf32>
          %19 = affine.apply #map1()[%arg5]
          %20 = affine.load %alloc_12[%arg4, %19] : memref<1x8xf32>
          %21 = arith.addf %18, %20 : f32
          %22 = math.tanh %21 : f32
          %23 = arith.mulf %16, %0 : f32
          %24 = arith.mulf %7, %22 : f32
          %25 = arith.addf %23, %24 : f32
          %26 = affine.apply #map2()[%arg5]
          %27 = affine.load %alloc_9[%arg4, %26] : memref<1x8xf32>
          %28 = affine.apply #map2()[%arg5]
          %29 = affine.load %alloc_12[%arg4, %28] : memref<1x8xf32>
          %30 = arith.addf %27, %29 : f32
          %31 = arith.subf %cst_0, %30 : f32
          %32 = math.exp %31 : f32
          %33 = arith.addf %32, %cst : f32
          %34 = arith.divf %cst, %33 : f32
          %35 = math.tanh %25 : f32
          %36 = arith.mulf %34, %35 : f32
          affine.store %25, %alloc_4[%arg4, %arg5] : memref<1x2xf32>
          affine.store %36, %alloc_3[%arg4, %arg5] : memref<1x2xf32>
          affine.store %36, %alloc[%arg3, %c0, %arg4, %arg5] : memref<1x1x1x2xf32>
        }
      }
    }
    "krnl.memcpy"(%alloc_1, %alloc_3, %c2_i64, %c0, %c0) : (memref<1x1x2xf32>, memref<1x2xf32>, i64, index, index) -> ()
    "krnl.memcpy"(%alloc_2, %alloc_4, %c2_i64, %c0, %c0) : (memref<1x1x2xf32>, memref<1x2xf32>, i64, index, index) -> ()
    return %alloc, %alloc_1, %alloc_2 : memref<1x1x1x2xf32>, memref<1x1x2xf32>, memref<1x1x2xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 3 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 10] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 8 , 10] , \22name\22 : \22in_b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 8 , 2] , \22name\22 : \22in_c\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 1 , 2] , \22name\22 : \22out_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 2] , \22name\22 : \22out_b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 2] , \22name\22 : \22out_c\22 }\0A\0A]\00"} : () -> ()
}
