#map = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map1 = affine_map<(d0) -> (-d0 + 2, 0)>
#map2 = affine_map<(d0) -> (-d0 + 30, 5)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0)>
#map4 = affine_map<(d0, d1) -> (d0 + d1 - 2)>
#map5 = affine_map<(d0, d1) -> (d0 - d1)>
module {
  func.func @main_graph(%arg0: memref<1x1x28x28xf32>, %arg1: memref<8x1x5x5xf32>, %arg2: memref<8xf32>) -> memref<1x8x28x28xf32> attributes {input_names = ["in_a", "in_b", "in_c"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x8x28x28xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 8 {
          %0 = affine.apply #map(%arg4, %arg5)
          affine.for %arg6 = 0 to 28 {
            affine.for %arg7 = 0 to 28 {
              %1 = affine.max #map1(%arg6)
              %2 = affine.min #map2(%arg6)
              %3 = affine.max #map1(%arg7)
              %4 = affine.min #map2(%arg7)
              %5 = arith.subi %2, %1 : index
              %6 = arith.subi %4, %3 : index
              %7 = arith.muli %5, %6 : index
              %alloc_0 = memref.alloc(%7) : memref<?xf32>
              %alloc_1 = memref.alloc(%7) : memref<?xf32>
              affine.for %arg8 = 0 to 1 {
                affine.for %arg9 = max #map1(%arg6) to min #map2(%arg6) {
                  affine.for %arg10 = max #map1(%arg7) to min #map2(%arg7) {
                    %11 = affine.apply #map3(%arg8)[%arg4]
                    %12 = affine.apply #map4(%arg9, %arg6)
                    %13 = affine.apply #map4(%arg10, %arg7)
                    %14 = affine.load %arg0[%arg3, %11, %12, %13] : memref<1x1x28x28xf32>
                    %15 = affine.load %arg1[%0, %arg8, %arg9, %arg10] : memref<8x1x5x5xf32>
                    %16 = affine.apply #map5(%arg10, %3)
                    %17 = affine.apply #map5(%arg9, %1)
                    %18 = arith.muli %6, %17 : index
                    %19 = arith.addi %16, %18 : index
                    memref.store %14, %alloc_0[%19] : memref<?xf32>
                    memref.store %15, %alloc_1[%19] : memref<?xf32>
                  }
                }
              }
              %8 = "zkML.dot-product"(%alloc_0, %alloc_1) : (memref<?xf32>, memref<?xf32>) -> f32
              memref.dealloc %alloc_0 : memref<?xf32>
              memref.dealloc %alloc_1 : memref<?xf32>
              %9 = affine.load %arg2[%0] : memref<8xf32>
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %alloc[%arg3, %0, %arg6, %arg7] : memref<1x8x28x28xf32>
            }
          }
        }
      }
    }
    return %alloc : memref<1x8x28x28xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 28 , 28] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [8 , 1 , 5 , 5] , \22name\22 : \22in_b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [8] , \22name\22 : \22in_c\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 8 , 28 , 28] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
