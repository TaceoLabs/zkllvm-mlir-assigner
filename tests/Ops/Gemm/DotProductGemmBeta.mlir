module {
  func.func @main_graph(%arg0: memref<8x16xf32>, %arg1: memref<16x12xf32>, %arg2: memref<12xf32>) -> memref<8x12xf32> attributes {input_names = ["in_a", "in_b", "in_c"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 5.439200e-01 : f32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<8x12xf32>
    %alloc_0 = memref.alloc() : memref<16xf32>
    %alloc_1 = memref.alloc() : memref<16xf32>
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 16 {
          %4 = affine.load %arg0[%arg3, %arg5] : memref<8x16xf32>
          %5 = affine.load %arg1[%arg5, %arg4] : memref<16x12xf32>
          affine.store %4, %alloc_0[%arg5] : memref<16xf32>
          affine.store %5, %alloc_1[%arg5] : memref<16xf32>
        }
        %0 = "zkML.dot-product"(%alloc_0, %alloc_1) : (memref<16xf32>, memref<16xf32>) -> f32
        %1 = affine.load %arg2[%arg4] : memref<12xf32>
        %2 = arith.mulf %1, %cst : f32
        %3 = arith.addf %0, %2 : f32
        affine.store %3, %alloc[%arg3, %arg4] : memref<8x12xf32>
      }
    }
    memref.dealloc %alloc_0 : memref<16xf32>
    memref.dealloc %alloc_1 : memref<16xf32>
    return %alloc : memref<8x12xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [8 , 16] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [16 , 12] , \22name\22 : \22in_b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [12] , \22name\22 : \22in_c\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [8 , 12] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
