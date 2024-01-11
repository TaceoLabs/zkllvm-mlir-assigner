#map = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map1 = affine_map<(d0) -> (-d0 + 2, 0)>
#map2 = affine_map<(d0) -> (-d0 + 5, 3)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0)>
#map4 = affine_map<(d0, d1) -> (d0 + d1 - 2)>
#map5 = affine_map<(d0, d1) -> (d0 - d1)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "convtransposesimple.mlir"} {
  func.func @main_graph(%arg0: memref<1x1x3x3xf32>, %arg1: memref<1x2x3x3xf32>) -> memref<1x2x5x5xf32> attributes {input_names = ["in_a", "in_b"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "krnl.global"() {name = "constant_1", shape = [3], value = dense<3> : tensor<3xi64>} : () -> memref<3xi64>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<3x3x1x2xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        affine.for %arg4 = 0 to 3 {
          affine.for %arg5 = 0 to 3 {
            %1 = affine.load %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x2x3x3xf32>
            affine.store %1, %alloc[%arg4, %arg5, %arg2, %arg3] : memref<3x3x1x2xf32>
          }
        }
      }
    }
    %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<3x3x1x2xf32>
    affine.for %arg2 = 0 to 3 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 1 {
          affine.for %arg5 = 0 to 2 {
            %1 = affine.load %0[%arg3] : memref<3xi64>
            %2 = arith.index_cast %1 : i64 to index
            %3 = arith.cmpi slt, %arg2, %2 : index
            %4 = arith.subi %2, %arg2 : index
            %5 = arith.subi %4, %c1 : index
            %6 = arith.select %3, %5, %arg2 : index
            %7 = memref.load %alloc[%6, %arg3, %arg4, %arg5] : memref<3x3x1x2xf32>
            affine.store %7, %alloc_0[%arg2, %arg3, %arg4, %arg5] : memref<3x3x1x2xf32>
          }
        }
      }
    }
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<3x3x1x2xf32>
    affine.for %arg2 = 0 to 3 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 1 {
          affine.for %arg5 = 0 to 2 {
            %1 = affine.load %0[%arg2] : memref<3xi64>
            %2 = arith.index_cast %1 : i64 to index
            %3 = arith.cmpi slt, %arg3, %2 : index
            %4 = arith.subi %2, %arg3 : index
            %5 = arith.subi %4, %c1 : index
            %6 = arith.select %3, %5, %arg3 : index
            %7 = memref.load %alloc_0[%arg2, %6, %arg4, %arg5] : memref<3x3x1x2xf32>
            affine.store %7, %alloc_1[%arg2, %arg3, %arg4, %arg5] : memref<3x3x1x2xf32>
          }
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 16 : i64} : memref<2x1x3x3xf32>
    affine.for %arg2 = 0 to 3 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 1 {
          affine.for %arg5 = 0 to 2 {
            %1 = affine.load %alloc_1[%arg2, %arg3, %arg4, %arg5] : memref<3x3x1x2xf32>
            affine.store %1, %alloc_2[%arg5, %arg4, %arg2, %arg3] : memref<2x1x3x3xf32>
          }
        }
      }
    }
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1x2x5x5xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        affine.for %arg4 = 0 to 2 {
          %1 = affine.apply #map(%arg3, %arg4)
          affine.for %arg5 = 0 to 5 {
            affine.for %arg6 = 0 to 5 {
              %2 = affine.max #map1(%arg5)
              %3 = affine.min #map2(%arg5)
              %4 = affine.max #map1(%arg6)
              %5 = affine.min #map2(%arg6)
              %6 = arith.subi %3, %2 : index
              %7 = arith.subi %5, %4 : index
              %8 = arith.muli %6, %7 : index
              %alloc_6 = memref.alloc(%8) : memref<?xf32>
              %alloc_7 = memref.alloc(%8) : memref<?xf32>
              %9 = arith.muli %6, %7 : index
              affine.for %arg7 = 0 to 1 {
                affine.for %arg8 = max #map1(%arg5) to min #map2(%arg5) {
                  affine.for %arg9 = max #map1(%arg6) to min #map2(%arg6) {
                    %11 = affine.apply #map3(%arg7)[%arg3]
                    %12 = affine.apply #map4(%arg8, %arg5)
                    %13 = affine.apply #map4(%arg9, %arg6)
                    %14 = affine.load %arg0[%arg2, %11, %12, %13] : memref<1x1x3x3xf32>
                    %15 = affine.load %alloc_2[%1, %arg7, %arg8, %arg9] : memref<2x1x3x3xf32>
                    %16 = affine.apply #map5(%arg9, %4)
                    %17 = affine.apply #map5(%arg8, %2)
                    %18 = arith.muli %7, %17 : index
                    %19 = arith.addi %16, %18 : index
                    %20 = arith.muli %9, %arg7 : index
                    %21 = arith.addi %19, %20 : index
                    memref.store %14, %alloc_6[%21] : memref<?xf32>
                    memref.store %15, %alloc_7[%21] : memref<?xf32>
                  }
                }
              }
              %10 = "zkML.dot-product"(%alloc_6, %alloc_7) : (memref<?xf32>, memref<?xf32>) -> f32
              memref.dealloc %alloc_6 : memref<?xf32>
              memref.dealloc %alloc_7 : memref<?xf32>
              affine.store %10, %alloc_3[%arg2, %1, %arg5, %arg6] : memref<1x2x5x5xf32>
            }
          }
        }
      }
    }
    %alloc_4 = memref.alloc() {alignment = 16 : i64} : memref<1x2x5x5xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        affine.for %arg4 = 0 to 5 {
          affine.for %arg5 = 0 to 5 {
            affine.store %cst, %alloc_4[%arg2, %arg3, %arg4, %arg5] : memref<1x2x5x5xf32>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        affine.for %arg4 = 0 to 5 {
          affine.for %arg5 = 0 to 5 {
            %1 = affine.load %alloc_3[%arg2, %arg3, %arg4, %arg5] : memref<1x2x5x5xf32>
            affine.store %1, %alloc_4[%arg2, %arg3, %arg4, %arg5] : memref<1x2x5x5xf32>
          }
        }
      }
    }
    %alloc_5 = memref.alloc() {alignment = 16 : i64} : memref<1x2x5x5xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        affine.for %arg4 = 0 to 5 {
          affine.for %arg5 = 0 to 5 {
            affine.store %cst, %alloc_5[%arg2, %arg3, %arg4, %arg5] : memref<1x2x5x5xf32>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        affine.for %arg4 = 0 to 5 {
          affine.for %arg5 = 0 to 5 {
            %1 = affine.load %alloc_4[%arg2, %arg3, %arg4, %arg5] : memref<1x2x5x5xf32>
            affine.store %1, %alloc_5[%arg2, %arg3, %arg4, %arg5] : memref<1x2x5x5xf32>
          }
        }
      }
    }
    return %alloc_5 : memref<1x2x5x5xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 3 , 3] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 2 , 3 , 3] , \22name\22 : \22in_b\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 2 , 5 , 5] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
