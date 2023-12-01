#map = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map1 = affine_map<(d0) -> (-d0 + 2, 0)>
#map2 = affine_map<(d0) -> (-d0 + 30, 5)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0)>
#map4 = affine_map<(d0, d1) -> (d0 + d1 - 2)>
#map5 = affine_map<(d0, d1) -> (d0 - d1)>
module {
  func.func @main_graph(%arg0: memref<1x1x28x28xf32>, %arg1: memref<8x1x5x5xf32>) -> memref<1x8x28x28xf32> attributes {input_names = ["in_a", "in_b"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x8x28x28xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        affine.for %arg4 = 0 to 8 {
          %0 = affine.apply #map(%arg3, %arg4)
          affine.for %arg5 = 0 to 28 {
            affine.for %arg6 = 0 to 28 {
              %1 = affine.max #map1(%arg5)
              %2 = affine.min #map2(%arg5)
              %3 = affine.max #map1(%arg6)
              %4 = affine.min #map2(%arg6)
              %5 = arith.subi %2, %1 : index
              %6 = arith.subi %4, %3 : index
              %7 = arith.muli %5, %6 : index
              %alloc_0 = memref.alloc(%7) : memref<?xf32>
              %alloc_1 = memref.alloc(%7) : memref<?xf32>
              affine.for %arg7 = 0 to 1 {
                affine.for %arg8 = max #map1(%arg5) to min #map2(%arg5) {
                  affine.for %arg9 = max #map1(%arg6) to min #map2(%arg6) {
                    %9 = affine.apply #map3(%arg7)[%arg3]
                    %10 = affine.apply #map4(%arg8, %arg5)
                    %11 = affine.apply #map4(%arg9, %arg6)
                    %12 = affine.load %arg0[%arg2, %9, %10, %11] : memref<1x1x28x28xf32>
                    %13 = affine.load %arg1[%0, %arg7, %arg8, %arg9] : memref<8x1x5x5xf32>
                    %14 = affine.apply #map5(%arg9, %3)
                    %15 = affine.apply #map5(%arg8, %1)
                    %16 = arith.muli %6, %15 : index
                    %17 = arith.addi %14, %16 : index
                    memref.store %12, %alloc_0[%17] : memref<?xf32>
                    memref.store %13, %alloc_1[%17] : memref<?xf32>
                  }
                }
              }
              %8 = "zkML.dot-product"(%alloc_0, %alloc_1) : (memref<?xf32>, memref<?xf32>) -> f32
              memref.dealloc %alloc_0 : memref<?xf32>
              memref.dealloc %alloc_1 : memref<?xf32>
              affine.store %8, %alloc[%arg2, %0, %arg5, %arg6] : memref<1x8x28x28xf32>
            }
          }
        }
      }
    }
    return %alloc : memref<1x8x28x28xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 28 , 28] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [8 , 1 , 5 , 5] , \22name\22 : \22in_b\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 8 , 28 , 28] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
