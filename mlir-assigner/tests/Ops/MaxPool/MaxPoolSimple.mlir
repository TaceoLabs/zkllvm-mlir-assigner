#map = affine_map<(d0) -> (0, d0 * 2)>
#map1 = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "maxpoolsimple.0.mlir"} {
  func.func @main_graph(%arg0: memref<1x1x28x28xf32>) -> memref<1x1x14x14xf32> attributes {input_names = ["in_a"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 0xFF800000 : f32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c28 = arith.constant 28 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1x14x14xf32>
    %alloca = memref.alloca() : memref<f32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            affine.store %cst, %alloca[] : memref<f32>
            %0 = affine.max #map(%arg3)
            %1 = affine.max #map(%arg4)
            affine.for %arg5 = 0 to min #map1(%arg3)[%c28, %c2, %c0, %c2, %c1] {
              affine.for %arg6 = 0 to min #map1(%arg4)[%c28, %c2, %c0, %c2, %c1] {
                %3 = arith.addi %arg5, %0 : index
                %4 = arith.addi %arg6, %1 : index
                %5 = memref.load %arg0[%arg1, %arg2, %3, %4] : memref<1x1x28x28xf32>
                %6 = affine.load %alloca[] : memref<f32>
                %7 = arith.cmpf ogt, %6, %5 : f32
                %8 = arith.select %7, %6, %5 : f32
                affine.store %8, %alloca[] : memref<f32>
              }
            }
            %2 = affine.load %alloca[] : memref<f32>
            affine.store %2, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x1x14x14xf32>
          }
        }
      }
    }
    return %alloc : memref<1x1x14x14xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 28 , 28] , \22name\22 : \22in_a\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 14 , 14] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
