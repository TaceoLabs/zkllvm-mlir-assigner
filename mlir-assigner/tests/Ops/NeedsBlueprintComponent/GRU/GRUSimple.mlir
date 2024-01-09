#map = affine_map<(d0) -> (d0 + 2)>
#map1 = affine_map<(d0) -> (d0 + 4)>
#map2 = affine_map<()[s0] -> (s0 + 2)>
#map3 = affine_map<()[s0] -> (s0 + 4)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "grusimple.mlir"} {
  func.func @main_graph(%arg0: memref<1x1x10xf32>, %arg1: memref<1x6x10xf32>, %arg2: memref<1x6x2xf32>) -> (memref<1x1x1x2xf32>, memref<1x1x2xf32>) attributes {input_names = ["in_a", "in_b", "in_c"], llvm.emit_c_interface, output_names = ["out_a", "out_b"]} {
    %c2_i64 = arith.constant 2 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x2xf32>
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x1x2xf32>
    %alloc_2 = memref.alloc() {alignment = 16 : i64} : memref<1x2xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 2 {
        affine.store %cst_0, %alloc_2[%arg3, %arg4] : memref<1x2xf32>
      }
    }
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [6, 10], strides: [10, 1] : memref<1x6x10xf32> to memref<6x10xf32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [6, 2], strides: [2, 1] : memref<1x6x2xf32> to memref<6x2xf32>
    %alloc_4 = memref.alloc() {alignment = 16 : i64} : memref<10x6xf32>
    affine.for %arg3 = 0 to 6 {
      affine.for %arg4 = 0 to 10 {
        %0 = affine.load %reinterpret_cast[%arg3, %arg4] : memref<6x10xf32>
        affine.store %0, %alloc_4[%arg4, %arg3] : memref<10x6xf32>
      }
    }
    %alloc_5 = memref.alloc() {alignment = 16 : i64} : memref<2x2xf32>
    %alloc_6 = memref.alloc() {alignment = 16 : i64} : memref<2x2xf32>
    %alloc_7 = memref.alloc() {alignment = 16 : i64} : memref<2x2xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.load %reinterpret_cast_3[%arg3, %arg4] : memref<6x2xf32>
        affine.store %0, %alloc_5[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.apply #map(%arg3)
        %1 = affine.load %reinterpret_cast_3[%0, %arg4] : memref<6x2xf32>
        affine.store %1, %alloc_6[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.apply #map1(%arg3)
        %1 = affine.load %reinterpret_cast_3[%0, %arg4] : memref<6x2xf32>
        affine.store %1, %alloc_7[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<2x2xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.load %alloc_5[%arg3, %arg4] : memref<2x2xf32>
        affine.store %0, %alloc_8[%arg4, %arg3] : memref<2x2xf32>
      }
    }
    %alloc_9 = memref.alloc() {alignment = 16 : i64} : memref<2x2xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.load %alloc_6[%arg3, %arg4] : memref<2x2xf32>
        affine.store %0, %alloc_9[%arg4, %arg3] : memref<2x2xf32>
      }
    }
    %alloc_10 = memref.alloc() {alignment = 16 : i64} : memref<2x2xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.load %alloc_7[%arg3, %arg4] : memref<2x2xf32>
        affine.store %0, %alloc_10[%arg4, %arg3] : memref<2x2xf32>
      }
    }
    affine.for %arg3 = 0 to 1 {
      %alloc_11 = memref.alloc() {alignment = 16 : i64} : memref<1x10xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 10 {
          %0 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<1x1x10xf32>
          affine.store %0, %alloc_11[%arg4, %arg5] : memref<1x10xf32>
        }
      }
      %alloc_12 = memref.alloc() {alignment = 16 : i64} : memref<1x6xf32>
      %alloc_13 = memref.alloc() : memref<10xf32>
      %alloc_14 = memref.alloc() : memref<10xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 6 {
          affine.for %arg6 = 0 to 10 {
            %1 = affine.load %alloc_11[%arg4, %arg6] : memref<1x10xf32>
            %2 = affine.load %alloc_4[%arg6, %arg5] : memref<10x6xf32>
            affine.store %1, %alloc_13[%arg6] : memref<10xf32>
            affine.store %2, %alloc_14[%arg6] : memref<10xf32>
          }
          %0 = "zkML.dot-product"(%alloc_13, %alloc_14) : (memref<10xf32>, memref<10xf32>) -> f32
          affine.store %0, %alloc_12[%arg4, %arg5] : memref<1x6xf32>
        }
      }
      memref.dealloc %alloc_13 : memref<10xf32>
      memref.dealloc %alloc_14 : memref<10xf32>
      %alloc_15 = memref.alloc() {alignment = 16 : i64} : memref<1x2xf32>
      %alloc_16 = memref.alloc() : memref<2xf32>
      %alloc_17 = memref.alloc() : memref<2xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 2 {
          affine.for %arg6 = 0 to 2 {
            %1 = affine.load %alloc_2[%arg4, %arg6] : memref<1x2xf32>
            %2 = affine.load %alloc_8[%arg6, %arg5] : memref<2x2xf32>
            affine.store %1, %alloc_16[%arg6] : memref<2xf32>
            affine.store %2, %alloc_17[%arg6] : memref<2xf32>
          }
          %0 = "zkML.dot-product"(%alloc_16, %alloc_17) : (memref<2xf32>, memref<2xf32>) -> f32
          affine.store %0, %alloc_15[%arg4, %arg5] : memref<1x2xf32>
        }
      }
      memref.dealloc %alloc_16 : memref<2xf32>
      memref.dealloc %alloc_17 : memref<2xf32>
      %alloc_18 = memref.alloc() {alignment = 16 : i64} : memref<1x2xf32>
      %alloc_19 = memref.alloc() : memref<2xf32>
      %alloc_20 = memref.alloc() : memref<2xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 2 {
          affine.for %arg6 = 0 to 2 {
            %1 = affine.load %alloc_2[%arg4, %arg6] : memref<1x2xf32>
            %2 = affine.load %alloc_9[%arg6, %arg5] : memref<2x2xf32>
            affine.store %1, %alloc_19[%arg6] : memref<2xf32>
            affine.store %2, %alloc_20[%arg6] : memref<2xf32>
          }
          %0 = "zkML.dot-product"(%alloc_19, %alloc_20) : (memref<2xf32>, memref<2xf32>) -> f32
          affine.store %0, %alloc_18[%arg4, %arg5] : memref<1x2xf32>
        }
      }
      memref.dealloc %alloc_19 : memref<2xf32>
      memref.dealloc %alloc_20 : memref<2xf32>
      %alloc_21 = memref.alloc() {alignment = 16 : i64} : memref<1x2xf32>
      %alloc_22 = memref.alloc() {alignment = 16 : i64} : memref<1x2xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 2 {
          %0 = affine.load %alloc_2[%arg4, %arg5] : memref<1x2xf32>
          %1 = affine.apply #map2()[%arg5]
          %2 = affine.load %alloc_12[%arg4, %1] : memref<1x6xf32>
          %3 = affine.load %alloc_18[%arg4, %arg5] : memref<1x2xf32>
          %4 = arith.addf %2, %3 : f32
          %5 = arith.subf %cst_0, %4 : f32
          %6 = math.exp %5 : f32
          %7 = arith.addf %6, %cst : f32
          %8 = arith.divf %cst, %7 : f32
          affine.store %8, %alloc_21[%arg4, %arg5] : memref<1x2xf32>
          %9 = arith.mulf %8, %0 : f32
          affine.store %9, %alloc_22[%arg4, %arg5] : memref<1x2xf32>
        }
      }
      %alloc_23 = memref.alloc() {alignment = 16 : i64} : memref<1x2xf32>
      %alloc_24 = memref.alloc() : memref<2xf32>
      %alloc_25 = memref.alloc() : memref<2xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 2 {
          affine.for %arg6 = 0 to 2 {
            %1 = affine.load %alloc_22[%arg4, %arg6] : memref<1x2xf32>
            %2 = affine.load %alloc_10[%arg6, %arg5] : memref<2x2xf32>
            affine.store %1, %alloc_24[%arg6] : memref<2xf32>
            affine.store %2, %alloc_25[%arg6] : memref<2xf32>
          }
          %0 = "zkML.dot-product"(%alloc_24, %alloc_25) : (memref<2xf32>, memref<2xf32>) -> f32
          affine.store %0, %alloc_23[%arg4, %arg5] : memref<1x2xf32>
        }
      }
      memref.dealloc %alloc_24 : memref<2xf32>
      memref.dealloc %alloc_25 : memref<2xf32>
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 2 {
          %0 = affine.load %alloc_2[%arg4, %arg5] : memref<1x2xf32>
          %1 = affine.load %alloc_12[%arg4, %arg5] : memref<1x6xf32>
          %2 = affine.load %alloc_15[%arg4, %arg5] : memref<1x2xf32>
          %3 = arith.addf %1, %2 : f32
          %4 = arith.subf %cst_0, %3 : f32
          %5 = math.exp %4 : f32
          %6 = arith.addf %5, %cst : f32
          %7 = arith.divf %cst, %6 : f32
          %8 = affine.apply #map3()[%arg5]
          %9 = affine.load %alloc_12[%arg4, %8] : memref<1x6xf32>
          %10 = affine.load %alloc_23[%arg4, %arg5] : memref<1x2xf32>
          %11 = arith.addf %9, %10 : f32
          %12 = math.tanh %11 : f32
          %13 = arith.subf %cst, %7 : f32
          %14 = arith.mulf %13, %12 : f32
          %15 = arith.mulf %7, %0 : f32
          %16 = arith.addf %14, %15 : f32
          affine.store %16, %alloc_2[%arg4, %arg5] : memref<1x2xf32>
          affine.store %16, %alloc[%arg3, %c0, %arg4, %arg5] : memref<1x1x1x2xf32>
        }
      }
    }
    "krnl.memcpy"(%alloc_1, %alloc_2, %c2_i64, %c0, %c0) : (memref<1x1x2xf32>, memref<1x2xf32>, i64, index, index) -> ()
    return %alloc, %alloc_1 : memref<1x1x1x2xf32>, memref<1x1x2xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 10] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 6 , 10] , \22name\22 : \22in_b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 6 , 2] , \22name\22 : \22in_c\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 1 , 2] , \22name\22 : \22out_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 2] , \22name\22 : \22out_b\22 }\0A\0A]\00"} : () -> ()
}
