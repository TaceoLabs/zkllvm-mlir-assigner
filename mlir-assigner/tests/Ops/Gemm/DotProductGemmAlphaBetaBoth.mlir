module {
  func.func @main_graph(%arg0: memref<1x196xf32>, %arg1: memref<196x128xf32>, %arg2: memref<128xf32>) -> memref<1x128xf32> attributes {input_names = ["in_a", "in_b", "in_c"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 1.020000e+01 : f32
    %cst_0 = arith.constant -3.450000e+00 : f32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<1x128xf32>
    %alloc_1 = memref.alloc() : memref<196xf32>
    %alloc_2 = memref.alloc() : memref<196xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 196 {
          %5 = affine.load %arg0[%arg3, %arg5] : memref<1x196xf32>
          %6 = affine.load %arg1[%arg5, %arg4] : memref<196x128xf32>
          affine.store %5, %alloc_1[%arg5] : memref<196xf32>
          affine.store %6, %alloc_2[%arg5] : memref<196xf32>
        }
        %0 = "zkML.dot-product"(%alloc_1, %alloc_2) : (memref<196xf32>, memref<196xf32>) -> f32
        %1 = arith.mulf %0, %cst_0 : f32
        %2 = affine.load %arg2[%arg4] : memref<128xf32>
        %3 = arith.mulf %2, %cst : f32
        %4 = arith.addf %1, %3 : f32
        affine.store %4, %alloc[%arg3, %arg4] : memref<1x128xf32>
      }
    }
    memref.dealloc %alloc_1 : memref<196xf32>
    memref.dealloc %alloc_2 : memref<196xf32>
    return %alloc : memref<1x128xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 196] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [196 , 128] , \22name\22 : \22in_b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [128] , \22name\22 : \22in_c\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 128] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
