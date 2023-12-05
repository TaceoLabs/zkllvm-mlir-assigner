module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "roundsimple.0.mlir"} {
  func.func @main_graph(%arg0: memref<1x10xf32>) -> memref<1x10xf32> attributes {input_names = ["in_a"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x10xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        %0 = affine.load %arg0[%arg1, %arg2] : memref<1x10xf32>
        %1 = math.floor %0 : f32
        %2 = arith.subf %0, %1 : f32
        %3 = arith.cmpf ogt, %2, %cst : f32
        %4 = arith.addf %1, %cst_1 : f32
        %5 = arith.select %3, %4, %1 : f32
        %6 = arith.mulf %1, %cst : f32
        %7 = math.floor %6 : f32
        %8 = arith.mulf %7, %cst_0 : f32
        %9 = arith.subf %1, %8 : f32
        %10 = arith.cmpf oeq, %9, %cst_1 : f32
        %11 = arith.addf %1, %cst_1 : f32
        %12 = arith.select %10, %11, %1 : f32
        %13 = arith.cmpf oeq, %2, %cst : f32
        %14 = arith.select %13, %12, %5 : f32
        affine.store %14, %alloc[%arg1, %arg2] : memref<1x10xf32>
      }
    }
    return %alloc : memref<1x10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22in_a\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
