module {
  func.func @main_graph(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) -> memref<16x16xf32> attributes {input_names = ["in_a", "in_b"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<16x16xf32>
    %alloca = memref.alloca() {alignment = 16 : i64} : memref<16xf32>
    %alloca_0 = memref.alloca() {alignment = 16 : i64} : memref<16xf32>
    affine.for %arg2 = 0 to 16 {
      affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 16 {
          %1 = affine.load %arg0[%arg2, %arg4] : memref<16x16xf32>
          %2 = affine.load %arg1[%arg4, %arg3] : memref<16x16xf32>
          affine.store %1, %alloca[%arg4] : memref<16xf32>
          affine.store %2, %alloca_0[%arg4] : memref<16xf32>
        }
        %0 = "zkML.dot-product"(%alloca, %alloca_0) : (memref<16xf32>, memref<16xf32>) -> f32
        affine.store %0, %alloc[%arg2, %arg3] : memref<16x16xf32>
      }
    }
    return %alloc : memref<16x16xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [16 , 16] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [16 , 16] , \22name\22 : \22in_b\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [16 , 16] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
