#map = affine_map<(d0, d1) -> (d0 * 16 + d1)>
#map1 = affine_map<(d0) -> (-d0 + 2, 0)>
#map2 = affine_map<(d0) -> (-d0 + 16, 5)>
#map3 = affine_map<(d0, d1) -> (d0 + d1 * 8)>
#map4 = affine_map<(d0, d1) -> (d0 + d1 - 2)>
#map5 = affine_map<(d0, d1) -> (d0 - d1)>
module {
  func.func @main_graph(%arg0: memref<1x8x14x14xf32>, %arg1: memref<16x8x5x5xf32>, %arg2: memref<16xf32>) -> memref<1x16x14x14xf32> attributes {input_names = ["in_a", "in_b", "in_c"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %c8 = arith.constant 8 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x16x14x14xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 16 {
          %0 = affine.apply #map(%arg4, %arg5)
          affine.for %arg6 = 0 to 14 {
            affine.for %arg7 = 0 to 14 {
              %1 = affine.max #map1(%arg6)
              %2 = affine.min #map2(%arg6)
              %3 = affine.max #map1(%arg7)
              %4 = affine.min #map2(%arg7)
              %5 = arith.subi %2, %1 : index
              %6 = arith.muli %5, %c8 : index
              %7 = arith.subi %4, %3 : index
              %8 = arith.muli %6, %7 : index
              %alloc_0 = memref.alloc(%8) : memref<?xf32>
              %alloc_1 = memref.alloc(%8) : memref<?xf32>
              %9 = arith.muli %5, %7 : index
              affine.for %arg8 = 0 to 8 {
                affine.for %arg9 = max #map1(%arg6) to min #map2(%arg6) {
                  affine.for %arg10 = max #map1(%arg7) to min #map2(%arg7) {
                    %13 = affine.apply #map3(%arg8, %arg4)
                    %14 = affine.apply #map4(%arg9, %arg6)
                    %15 = affine.apply #map4(%arg10, %arg7)
                    %16 = affine.load %arg0[%arg3, %13, %14, %15] : memref<1x8x14x14xf32>
                    %17 = affine.load %arg1[%0, %arg8, %arg9, %arg10] : memref<16x8x5x5xf32>
                    %18 = affine.apply #map5(%arg10, %3)
                    %19 = affine.apply #map5(%arg9, %1)
                    %20 = arith.muli %7, %19 : index
                    %21 = arith.addi %18, %20 : index
                    %22 = arith.muli %9, %arg8 : index
                    %23 = arith.addi %21, %22 : index
                    memref.store %16, %alloc_0[%23] : memref<?xf32>
                    memref.store %17, %alloc_1[%23] : memref<?xf32>
                  }
                }
              }
              %10 = "zkML.dot-product"(%alloc_0, %alloc_1) : (memref<?xf32>, memref<?xf32>) -> f32
              memref.dealloc %alloc_0 : memref<?xf32>
              memref.dealloc %alloc_1 : memref<?xf32>
              %11 = affine.load %arg2[%0] : memref<16xf32>
              %12 = arith.addf %10, %11 : f32
              affine.store %12, %alloc[%arg3, %0, %arg6, %arg7] : memref<1x16x14x14xf32>
            }
          }
        }
      }
    }
    return %alloc : memref<1x16x14x14xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 8 , 14 , 14] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [16 , 8 , 5 , 5] , \22name\22 : \22in_b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [16] , \22name\22 : \22in_c\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 16 , 14 , 14] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
