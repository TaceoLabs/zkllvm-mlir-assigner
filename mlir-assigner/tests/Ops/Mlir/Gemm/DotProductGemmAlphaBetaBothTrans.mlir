module {
  func.func @main_graph(%arg0: memref<16x4xf32>, %arg1: memref<12x16xf32>, %arg2: memref<12xf32>) -> memref<4x12xf32> attributes {input_names = ["in_a", "in_b", "in_c"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant -1.303520e-01 : f32
    %cst_0 = arith.constant 3.450000e+00 : f32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<4x12xf32>
    %alloc_1 = memref.alloc() : memref<16xf32>
    %alloc_2 = memref.alloc() : memref<16xf32>
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 16 {
          %5 = affine.load %arg0[%arg5, %arg3] : memref<16x4xf32>
          %6 = affine.load %arg1[%arg4, %arg5] : memref<12x16xf32>
          affine.store %5, %alloc_1[%arg5] : memref<16xf32>
          affine.store %6, %alloc_2[%arg5] : memref<16xf32>
        }
        %0 = "zkML.dot-product"(%alloc_1, %alloc_2) : (memref<16xf32>, memref<16xf32>) -> f32
        %1 = arith.mulf %0, %cst_0 : f32
        %2 = affine.load %arg2[%arg4] : memref<12xf32>
        %3 = arith.mulf %2, %cst : f32
        %4 = arith.addf %1, %3 : f32
        affine.store %4, %alloc[%arg3, %arg4] : memref<4x12xf32>
      }
    }
    memref.dealloc %alloc_1 : memref<16xf32>
    memref.dealloc %alloc_2 : memref<16xf32>
    return %alloc : memref<4x12xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [16 , 4] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [12 , 16] , \22name\22 : \22in_b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [12] , \22name\22 : \22in_c\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [4 , 12] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
