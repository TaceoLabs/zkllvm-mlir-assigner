module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "hardmaxsimple.mlir"} {
  func.func @main_graph(%arg0: memref<1x10x20x30xf32>) -> memref<1x10x20x30xf32> attributes {input_names = ["in_a"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x10x20x30xf32>
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x10x20x1xindex>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        affine.for %arg3 = 0 to 20 {
          affine.for %arg4 = 0 to 1 {
            affine.store %c0, %alloc_1[%arg1, %arg2, %arg3, %arg4] : memref<1x10x20x1xindex>
          }
        }
      }
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        affine.for %arg3 = 0 to 20 {
          affine.for %arg4 = 0 to 30 {
            %0 = affine.load %alloc_1[%arg1, %arg2, %arg3, %c0] : memref<1x10x20x1xindex>
            %1 = memref.load %arg0[%arg1, %arg2, %arg3, %0] : memref<1x10x20x30xf32>
            %2 = affine.load %arg0[%arg1, %arg2, %arg3, %arg4] : memref<1x10x20x30xf32>
            %3 = arith.cmpf ogt, %2, %1 : f32
            scf.if %3 {
              affine.store %arg4, %alloc_1[%arg1, %arg2, %arg3, %c0] : memref<1x10x20x1xindex>
            }
          }
        }
      }
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        affine.for %arg3 = 0 to 20 {
          affine.for %arg4 = 0 to 30 {
            %0 = affine.load %alloc_1[%arg1, %arg2, %arg3, %c0] : memref<1x10x20x1xindex>
            %1 = arith.cmpi eq, %0, %arg4 : index
            scf.if %1 {
              affine.store %cst_0, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x10x20x30xf32>
            } else {
              affine.store %cst, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x10x20x30xf32>
            }
          }
        }
      }
    }
    return %alloc : memref<1x10x20x30xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10 , 20 , 30] , \22name\22 : \22in_a\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10 , 20 , 30] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
