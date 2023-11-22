# build the mlir-assigner
build: 
  make -C build/ -j 12

# setsup the build folder
setup-build:
  mkdir -p build
  cmake -DMLIR_DIR=${MLIR_DIR} -DONNX_USE_PROTOBUF_SHARED_LIBS=ON -Bbuild -S.

# runs only the small model tests (single onnx operations)
run-fast-tests: build
  python tests/run.py --fast

# runs all tests. This will take A LONG time
run-tests: build
  python tests/run.py --fast

# runs the basic mnist model and prints the output to stdout
run-basic-mnist: build
  ./build/src/mlir-assigner -b tests/Models/BasicMnist/basic_mnist.mlir -i tests/Models/BasicMnist/basic_mnist.json -e pallas -c circuit -t table --print_circuit_output --check
  rm circuit
  rm table
