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

- `zkml-onnx-compiler` A compiler that lowers `.onnx` files. (name is subject to
  change)
- `mlir-assigner` An assigner that interprets `.mlir` dialect files and produces a plonkish circuit
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
cmake -DMLIR_DIR=${MLIR_DIR} ..
make -j
```

You can find the binaries in the `build/bin` folder.

### Build on Ubuntu 22.04

We stress again to use the preferred approach, the `nix` devshell. If you
still want to built on your native machine, follow these steps.

1. Download the source:

```bash
git clone --recurse-submodules -j8 git@github.com:TaceoLabs/zkllvm-mlir-assigner.git && cd zkllvm-mlir-assigner
```

2. One of our dependencies is ONNX-MLIR (`libs/onnx-mlir`).
   [Here](https://github.com/onnx/onnx-mlir/tree/a04f518c1b0b8e4971d554c399bb54efc00b81db#setting-up-onnx-mlir-directly)
   you can find the requirements and built instructions for ONNX-MLIR. You do
   not have to built it, but every thing needs to be setup correctly. This
   includes a build tree of MLIR (not ONNX-MLIR) as part of the LLVM project, as
   seen
   [here](https://github.com/onnx/onnx-mlir/blob/a04f518c1b0b8e4971d554c399bb54efc00b81db/docs/BuildOnLinuxOSX.md).

3. At this point your environment variable `MLIR_DIR` should point to a MLIR cmake module. Verify that by writing
```bash
echo $MLIR_DIR
```

4. Setup cmake and build
```bash
mkdir build && cd build
cmake -DMLIR_DIR=${MLIR_DIR} ..
make -j zkml-onnx-compiler mlir-assigner
```

## Testing

To test your build, have a look in the [test folder](mlir-assigner/tests).

## Example

This section shows how to use the zkML frontend for zkLLVM. In this example, we
guide you through every step to proof the
[CNN-MNIST Model](https://github.com/onnx/models/tree/ddbbd1274c8387e3745778705810c340dea3d8c7/validated/vision/classification/mnist)
from the ONNX model zoo on Ubuntu 22.04.

We expect that you already built the two binaries `build/bin/zk-ml-opt` and
`build/bin/mlir-assigner` from source or obtained them in another way. If not,
follow the [build instructions](#build).

1. **Setup:** We start by creating an empty folder here we place our binaries
   and our model. So when you built from source we do:

```bash
mkdir CNN-Mnist
cp build/bin/mlir-assigner CNN-Mnist && cp build/bin/zkml-onnx-compiler CNN-Mnist
cd CNN-Mnist
wget https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx
```

In case you have another model you want to proof, use that instead of the
CNN-Mnist model.

2. **Compile ONNX to MLIR:** Having your pre-trained model at place, we use the
   `zkml-onnx-compiler` to compile the model to `.mlir`.
   ![compile](docs/pics/GitHubReadMeStep2.svg) You can do this by calling the
   `zkml-onnx-compiler`:

```bash
./zkml-onnx-compiler mnist-12.onnx -i mnist-12.mlir
```

> The `zkml-onnx-compiler` can also lower the model to different IRs. Have a
> look by adding the `--help` flag.

The emitted `.mlir` consists of the dialects defined by ONNX-MLIR and an
additional dialect defined by this project with the namespace `zkML`. This
dialect defines operations which need special handling in contrast to the
default lowering provided by ONNX-MLIR, e.g., as seen during the lowering of
[MatMuls](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul),
[Gemms](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm), or
[Convolutions](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv).
Natively lowering matrix multiplications leads to polluted traces with a lot of
additions and multiplications. For that we introduced the operation
`zkml.Dot-Product` which improves performance drastically.

3. **Prepare your input:** The `mlir-assigner` expects the input for inference
   in `json` format. Have a look at e.g.,
   [the input to this small model](mlir-assigner/tests/Ops/Onnx/Add/AddSimple.json)
   to see how you should prepare your input. For this example, we have an input
   file prepared. Therefore, we just copy it to our folder:

```bash
cp ../mlir-assigner/tests/Models/ConvMnist/mnist-12.json .
```

![inference](docs/pics/GitHubReadMeStep3.svg) 4. **Perform inference and assign
circuit:** The next step may take some time depending on your model. We have to
perform inference of the model and the provided input within the plonkish
arithmetization of the proof system. We do this by calling:

```bash
./mlir-assigner -b mnist-12.mlir -i mnist-12.json -e pallas -c circuit.crt -t assignment.tbl --print_circuit_output --check
```

> Again, have a look on the configurations of the `mlir-assigner` by adding the
> `--help` flag.

You can find the unassigned circuit in the file `circuit.crt` and the assignment
in `assignment.tbl`, as you are used to from `zkLLVM`. We refer to the
documentation from [zkLLVM](https://github.com/NilFoundation/zkLLVM#usage) on
how to produce proofs with `Placeholder` and/or publish your circuits to the
proof market.

## Disclaimer

This is **experimental software** and is provided on an "as is" and "as
available" basis. We do **not give any warranties** and will **not be liable for
any losses** incurred through any use of this code base.
