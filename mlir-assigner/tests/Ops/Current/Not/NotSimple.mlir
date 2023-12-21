module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "notsimple.mlir"} {
  func.func @main_graph(%arg0: memref<1x10xi1>) -> memref<1x10xi1> attributes {input_names = ["in_a"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %true = arith.constant true
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x10xi1>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        %0 = affine.load %arg0[%arg1, %arg2] : memref<1x10xi1>
        %1 = arith.xori %0, %true : i1
        affine.store %1, %alloc[%arg1, %arg2] : memref<1x10xi1>
      }
    }
    return %alloc : memref<1x10xi1>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22i1\22 , \22dims\22 : [1 , 10] , \22name\22 : \22in_a\22 }\0A\0A]\00@[   { \22type\22 : \22i1\22 , \22dims\22 : [1 , 10] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
