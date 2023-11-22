# Experimental MLIR Assigner

## Build

To build on ubuntu 22.04, you will need to install protobuf 3.20.3 from source, since the version in the Ubuntu repositores is too old.
You will also need a build version of zkllvm, configured with the following CMAKE options:

```bash
cmake -G "Unix Makefiles" -B ${ZKLLVM_BUILD:-build} -DCMAKE_BUILD_TYPE=Release -DCIRCUIT_ASSEMBLY_OUTPUT=TRUE -DLLVM_ENABLE_PROJECTS="clang;mlir" -DLLVM_TARGETS_TO_BUILD="Assigner;X86" -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_LIBEDIT=OFF .
make -C ${ZKLLVM_BUILD:-build} all -j8
```

Then, we can configure this project and point the MLIR_DIR variable to the finished build folder of zkllvm above.

```bash
mkdir build && cd build
cmake -DMLIR_DIR=../zkllvm/build/lib/cmake/mlir ..
```

## Build in nix devshell

The included nix flake sets up a devshell with all dependencies for building the MLIR assigner.
Since it also fetches and builds the bundled LLVM in zkllvm, this may take some time, so we recommend to pass `--cores 16` or similar to the initial invocation of `nix develop`.

```bash
nix develop # or
nix develop --cores 16 #allow the first-time build to use more cores for building the deps
# in the devshell
mkdir build && cd build
cmake -DMLIR_DIR=$MLIR_DIR ..
make -j
```

## Testing

To run the unit-tests for simple operations run:

```bash
python tests/run.py --fast
```

To run an example for a basic MNIST model:

```bash
./build/src/mlir-assigner -b tests/Models/BasicMnist/basic_mnist.mlir -i tests/Models/BasicMnist/basic_mnist.json -e pallas -c circuit -t table --print_circuit_output --check
```
