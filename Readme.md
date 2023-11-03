# Build

To build on ubuntu 22.04

```bash
sudo apt-get install libprotobuf-dev protobuf-compiler
mkdir build && cd build
cmake -DMLIR_DIR=~/repos/taceolabs/nil/zkllvm/build/lib/cmake/mlir -DONNX_USE_PROTOBUF_SHARED_LIBS=ON ..
```
