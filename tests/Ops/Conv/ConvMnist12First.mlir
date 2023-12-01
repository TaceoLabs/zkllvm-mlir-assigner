#map = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map1 = affine_map<(d0) -> (-d0 + 2, 0)>
#map2 = affine_map<(d0) -> (-d0 + 30, 5)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0)>
#map4 = affine_map<(d0, d1) -> (d0 + d1 - 2)>
module {
  func.func @main_graph(%arg0: memref<1x1x28x28xf32>, %arg1: memref<8x1x5x5xf32>) -> memref<1x8x28x28xf32> attributes {input_names = ["in_a", "in_b"], llvm.emit_c_interface, output_names = ["out_a"]} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x8x28x28xf32>
    %alloca = memref.alloca() : memref<f32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        affine.for %arg4 = 0 to 8 {
          %0 = affine.apply #map(%arg3, %arg4)
          affine.for %arg5 = 0 to 28 {
            affine.for %arg6 = 0 to 28 {
              affine.store %cst, %alloca[] : memref<f32>
              affine.for %arg7 = 0 to 1 {
                affine.for %arg8 = max #map1(%arg5) to min #map2(%arg5) {
                  affine.for %arg9 = max #map1(%arg6) to min #map2(%arg6) {
                    %2 = affine.apply #map3(%arg7)[%arg3]
                    %3 = affine.apply #map4(%arg8, %arg5)
                    %4 = affine.apply #map4(%arg9, %arg6)
                    %5 = affine.load %arg0[%arg2, %2, %3, %4] : memref<1x1x28x28xf32>
                    %6 = affine.load %arg1[%0, %arg7, %arg8, %arg9] : memref<8x1x5x5xf32>
                    %7 = affine.load %alloca[] : memref<f32>
                    %8 = arith.mulf %5, %6 : f32
                    %9 = arith.addf %7, %8 : f32
                    affine.store %9, %alloca[] : memref<f32>
                  }
                }
              }
              %1 = affine.load %alloca[] : memref<f32>
              affine.store %1, %alloc[%arg2, %0, %arg5, %arg6] : memref<1x8x28x28xf32>
            }
          }
        }
      }
    }
    return %alloc : memref<1x8x28x28xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 28 , 28] , \22name\22 : \22in_a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [8 , 1 , 5 , 5] , \22name\22 : \22in_b\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 8 , 28 , 28] , \22name\22 : \22out_a\22 }\0A\0A]\00"} : () -> ()
}
