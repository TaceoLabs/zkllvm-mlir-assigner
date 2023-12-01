#map = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map1 = affine_map<(d0) -> (-d0 + 2, 0)>
#map2 = affine_map<(d0) -> (-d0 + 30, 5)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0)>
#map4 = affine_map<(d0, d1) -> (d0 + d1 - 2)>
module {
  func.func @main_graph(%arg0: memref<1x1x28x28xf32>, %arg1: memref<8x1x5x5xf32>, %arg2: memref<8xf32>) -> memref<1x8x28x28xf32> attributes {input_names = ["in_a", "in_b", "in_c"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x8x28x28xf32>
    %alloca = memref.alloca() : memref<f32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 8 {
          %0 = affine.apply #map(%arg4, %arg5)
          affine.for %arg6 = 0 to 28 {
            affine.for %arg7 = 0 to 28 {
              affine.store %cst, %alloca[] : memref<f32>
              affine.for %arg8 = 0 to 1 {
                affine.for %arg9 = max #map1(%arg6) to min #map2(%arg6) {
                  affine.for %arg10 = max #map1(%arg7) to min #map2(%arg7) {
                    %4 = affine.apply #map3(%arg8)[%arg4]
                    %5 = affine.apply #map4(%arg9, %arg6)
                    %6 = affine.apply #map4(%arg10, %arg7)
                    %7 = affine.load %arg0[%arg3, %4, %5, %6] : memref<1x1x28x28xf32>
                    %8 = affine.load %arg1[%0, %arg8, %arg9, %arg10] : memref<8x1x5x5xf32>
                    %9 = affine.load %alloca[] : memref<f32>
                    %10 = arith.mulf %7, %8 : f32
                    %11 = arith.addf %9, %10 : f32
                    affine.store %11, %alloca[] : memref<f32>
                  }
                }
              }
              %1 = affine.load %alloca[] : memref<f32>
              %2 = affine.load %arg2[%0] : memref<8xf32>
              %3 = arith.addf %1, %2 : f32
              affine.store %3, %alloc[%arg3, %0, %arg6, %arg7] : memref<1x8x28x28xf32>
            }
          }
        }
      }
    }
    return %alloc : memref<1x8x28x28xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 28 , 28] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [8 , 1 , 5 , 5] , \22name\22 : \22in_b\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [8] , \22name\22 : \22in_c\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 8 , 28 , 28] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
