# ONNX zkML Frontend for zkLLVM

[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/TACEO_IO)](https://twitter.com/TACEO_IO)

This project extends the [zkLLVM](https://github.com/NilFoundation/zkLLVM)
compiler with an [ONNX](https://github.com/onnx/onnx#readme) frontend,
leveraging the
[ONNX-MLIR compiler technology](https://github.com/onnx/onnx-mlir). It allows
developers to compile and assign ZK circuits from their pre-trained `onnx`
models and proof them with the `Placeholder` proof system (see `zkLLVM`
documentation).

This project builds two binaries, namely:

- `zk-ml-opt` A compiler that lowers `.onnx` files. (name is subject to change)
- `mlir-assigner` A VM that interprets `.mlir` and produces a plonkish circuit
  and an assigned table.

We list the currently supported ONNX operations and limitations
[here](mlir-assigner/tests/README.md).

## Build

We heavily encourage to use the `nix` devshell when working with, or building
from source. It is possible to build the project on Linux (tested with Ubuntu
22.04), but the build process can be involved.

### Build in nix devshell

The included nix flake sets up a devshell with all dependencies for building the
`mlir-assigner`. Since it also fetches and builds the bundled LLVM in zkLLVM,
this may take some time, so we recommend to pass `--cores 16` or similar to the
initial invocation of `nix develop`.

```bash
nix develop # or
nix develop --cores 16 #allow the first-time build to use more cores for building the deps
# in the devshell
mkdir build && cd build
cmake -DMLIR_DIR=${MLIR_DIR} -DONNX_USE_PROTOBUF_SHARED_LIBS=ON ..
make -j
```

You can find the binaries in the `build/bin` folder.

### Build on Ubuntu 22.04

TODO

## Testing

To test your build, have a look in the [test folder](tests).

## Example

This section shows how to use the zkML frontend for zkLLVM. In this example, we
guide you through every step to proof the
[CNN-MNIST Model](https://github.com/onnx/models/tree/main/vision/classification/mnist)
found in `tests/Models/ConvMnist/mnist-12.onnx`.

We expect that you already built the two binaries `build/bin/zk-ml-opt` and
`build/bin/mlir-assigner` from source or obtained them in another way. If not,
follow the [build instructions](#build).

1. Lorem ipsum dolor sit amet, qui minim labore adipisicing minim sint cillum
   sint consectetur cupidatat.

2. Lorem ipsum dolor sit amet, officia excepteur ex fugiat reprehenderit enim
   labore culpa sint ad nisi Lorem pariatur mollit ex esse exercitation amet.
   Nisi anim cupidatat excepteur officia. Reprehenderit nostrud nostrud ipsum
   Lorem est aliquip amet voluptate voluptate dolor minim nulla est proident.
   Nostrud officia pariatur ut officia. Sit irure elit esse ea nulla sunt ex
   occaecat reprehenderit commodo officia dolor Lorem duis laboris cupidatat
   officia voluptate. Culpa proident adipisicing id nulla nisi laboris ex in
   Lorem sunt duis officia eiusmod. Aliqua reprehenderit commodo ex non
   excepteur duis sunt velit enim. Voluptate laboris sint cupidatat ullamco ut
   ea consectetur et est culpa et culpa duis.

3. Lorem ipsum dolor sit amet, officia excepteur ex fugiat reprehenderit enim
   labore culpa sint ad nisi Lorem pariatur mollit ex esse exercitation amet.
   Nisi anim cupidatat excepteur officia. Reprehenderit nostrud nostrud ipsum
   Lorem est aliquip amet voluptate voluptate dolor minim nulla est proident.
   Nostrud officia pariatur ut officia. Sit irure elit esse ea nulla sunt ex
   occaecat reprehenderit commodo officia dolor Lorem duis laboris cupidatat
   officia voluptate. Culpa proident adipisicing id nulla nisi laboris ex in
   Lorem sunt duis officia eiusmod. Aliqua reprehenderit commodo ex non
   excepteur duis sunt velit enim. Voluptate laboris sint cupidatat ullamco ut
   ea consectetur et est culpa et culpa duis.

To run an example for a basic MNIST model:

```bash
./build/src/mlir-assigner -b tests/Models/BasicMnist/basic_mnist.mlir -i tests/Models/BasicMnist/basic_mnist.json -e pallas -c circuit -t table --print_circuit_output --check
```
