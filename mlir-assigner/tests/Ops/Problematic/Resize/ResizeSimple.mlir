module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-linux-gnu", "onnx-mlir.symbol-postfix" = "resizesimple.mlir"} {
  func.func @main_graph(%arg0: memref<1x128x56x56xf32>) -> memref<1x128x112x112xf32> attributes {input_names = ["in_a", "in_b"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %c127 = arith.constant 127 : index
    %c55 = arith.constant 55 : index
    %cst = arith.constant 4.999990e-01 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 2.000000e+00 : f32
    %c56 = arith.constant 56 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x128x112x112xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 112 {
          affine.for %arg5 = 0 to 112 {
            %0 = arith.index_cast %arg2 : index to i64
            %1 = arith.sitofp %0 : i64 to f32
            %2 = arith.addf %1, %cst_0 : f32
            %3 = arith.subf %2, %cst_0 : f32
            %4 = arith.addf %3, %cst : f32
            %5 = arith.fptosi %4 : f32 to i64
            %6 = arith.cmpi slt, %5, %c0_i64 : i64
            %7 = arith.select %6, %c0_i64, %5 : i64
            %8 = arith.index_cast %7 : i64 to index
            %9 = arith.cmpi slt, %8, %c1 : index
            %10 = arith.select %9, %8, %c0 : index
            %11 = arith.index_cast %arg3 : index to i64
            %12 = arith.sitofp %11 : i64 to f32
            %13 = arith.addf %12, %cst_0 : f32
            %14 = arith.subf %13, %cst_0 : f32
            %15 = arith.addf %14, %cst : f32
            %16 = arith.fptosi %15 : f32 to i64
            %17 = arith.cmpi slt, %16, %c0_i64 : i64
            %18 = arith.select %17, %c0_i64, %16 : i64
            %19 = arith.index_cast %18 : i64 to index
            %20 = arith.cmpi slt, %19, %c128 : index
            %21 = arith.select %20, %19, %c127 : index
            %22 = arith.index_cast %arg4 : index to i64
            %23 = arith.sitofp %22 : i64 to f32
            %24 = arith.addf %23, %cst_0 : f32
            %25 = arith.divf %24, %cst_1 : f32
            %26 = arith.subf %25, %cst_0 : f32
            %27 = arith.addf %26, %cst : f32
            %28 = arith.fptosi %27 : f32 to i64
            %29 = arith.cmpi slt, %28, %c0_i64 : i64
            %30 = arith.select %29, %c0_i64, %28 : i64
            %31 = arith.index_cast %30 : i64 to index
            %32 = arith.cmpi slt, %31, %c56 : index
            %33 = arith.select %32, %31, %c55 : index
            %34 = arith.index_cast %arg5 : index to i64
            %35 = arith.sitofp %34 : i64 to f32
            %36 = arith.addf %35, %cst_0 : f32
            %37 = arith.divf %36, %cst_1 : f32
            %38 = arith.subf %37, %cst_0 : f32
            %39 = arith.addf %38, %cst : f32
            %40 = arith.fptosi %39 : f32 to i64
            %41 = arith.cmpi slt, %40, %c0_i64 : i64
            %42 = arith.select %41, %c0_i64, %40 : i64
            %43 = arith.index_cast %42 : i64 to index
            %44 = arith.cmpi slt, %43, %c56 : index
            %45 = arith.select %44, %43, %c55 : index
            %46 = memref.load %arg0[%10, %21, %33, %45] : memref<1x128x56x56xf32>
            affine.store %46, %alloc[%arg2, %arg3, %arg4, %arg5] : memref<1x128x112x112xf32>
          }
        }
      }
    }
    return %alloc : memref<1x128x112x112xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 128 , 56 , 56] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [] , \22name\22 : \22in_b\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 128 , 112 , 112] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
