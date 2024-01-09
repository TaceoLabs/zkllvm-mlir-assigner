#map = affine_map<(d0) -> (0, d0 - 1)>
#map1 = affine_map<(d0) -> (5, d0 + 2)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "lrnsimple.mlir"} {
  func.func @main_graph(%arg0: memref<5x5x5x5xf32>) -> memref<5x5x5x5xf32> attributes {input_names = ["in_a"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 3.33333337E-5 : f32
    %cst_2 = arith.constant 7.500000e-01 : f32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<5x5x5x5xf32>
    affine.for %arg1 = 0 to 5 {
      affine.for %arg2 = 0 to 5 {
        affine.for %arg3 = 0 to 5 {
          affine.for %arg4 = 0 to 5 {
            %alloc_3 = memref.alloc() : memref<f32>
            affine.store %cst, %alloc_3[] : memref<f32>
            affine.for %arg5 = max #map(%arg2) to min #map1(%arg2) {
              %6 = affine.load %arg0[%arg1, %arg5, %arg3, %arg4] : memref<5x5x5x5xf32>
              %7 = arith.mulf %6, %6 : f32
              %8 = affine.load %alloc_3[] : memref<f32>
              %9 = arith.addf %8, %7 : f32
              affine.store %9, %alloc_3[] : memref<f32>
            }
            %0 = affine.load %arg0[%arg1, %arg2, %arg3, %arg4] : memref<5x5x5x5xf32>
            %1 = affine.load %alloc_3[] : memref<f32>
            %2 = arith.mulf %1, %cst_1 : f32
            %3 = arith.addf %2, %cst_0 : f32
            %4 = math.powf %3, %cst_2 : f32
            %5 = arith.divf %0, %4 : f32
            affine.store %5, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<5x5x5x5xf32>
          }
        }
      }
    }
    return %alloc : memref<5x5x5x5xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [5 , 5 , 5 , 5] , \22name\22 : \22in_a\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [5 , 5 , 5 , 5] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
