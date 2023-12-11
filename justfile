# build the mlir-assigner
build: 
  make -C build/ -j 12

# setsup the build folder
setup-build:
  mkdir -p build
  cmake -DMLIR_DIR=${MLIR_DIR} -DONNX_USE_PROTOBUF_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -Bbuild -S.

# runs only the small model tests (single onnx operations)
run-fast-tests: build
  python mlir-assigner/tests/run.py --fast

# runs all tests. This will take A LONG time
run-tests: build
  python mlir-assigner/tests/run.py 

# runs the basic mnist model and prints the output to stdout
run-basic-mnist: build
  ./build/bin/mlir-assigner -b mlir-assigner/tests/Models/BasicMnist/BasicMnist.mlir -i mlir-assigner/tests/Models/BasicMnist/BasicMnist.json -e pallas -c circuit -t table --print_circuit_output --check
  rm circuit
  rm table

# runs the basic mnist model and prints the output to stdout
run-basic-mnist-dot: build
  ./build/bin/mlir-assigner -b mlir-assigner/tests/Models/BasicMnist/DotProductBasicMnist.mlir -i mlir-assigner/tests/Models/BasicMnist/DotProductBasicMnist.json -e pallas -c circuit -t table --print_circuit_output --check
  rm circuit
  rm table
