module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "cumsumsimple.mlir"} {
  func.func @main_graph(%arg0: memref<1x25xf32>) -> memref<1x25xf32> attributes {input_names = ["in_a"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 0.693147182 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x25xf32>
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x25xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 25 {
        %0 = affine.load %arg0[%arg1, %arg2] : memref<1x25xf32>
        affine.store %0, %alloc_1[%arg1, %arg2] : memref<1x25xf32>
      }
    }
    affine.for %arg1 = 0 to 5 {
      %0 = arith.index_cast %arg1 : index to i64
      %1 = arith.sitofp %0 : i64 to f32
      %2 = arith.mulf %1, %cst : f32
      %3 = math.exp %2 : f32
      %4 = arith.fptosi %3 : f32 to i64
      %5 = arith.index_cast %4 : i64 to index
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 25 {
          %6 = affine.load %alloc_1[%arg2, %arg3] : memref<1x25xf32>
          %7 = arith.subi %arg3, %5 : index
          %8 = arith.cmpi sge, %7, %c0 : index
          %9 = arith.select %8, %7, %arg3 : index
          %10 = memref.load %alloc_1[%arg2, %9] : memref<1x25xf32>
          %11 = arith.select %8, %10, %cst_0 : f32
          %12 = arith.addf %6, %11 : f32
          affine.store %12, %alloc[%arg2, %arg3] : memref<1x25xf32>
        }
      }
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 25 {
          %6 = affine.load %alloc[%arg2, %arg3] : memref<1x25xf32>
          affine.store %6, %alloc_1[%arg2, %arg3] : memref<1x25xf32>
        }
      }
    }
    return %alloc : memref<1x25xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 25] , \22name\22 : \22in_a\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 25] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
