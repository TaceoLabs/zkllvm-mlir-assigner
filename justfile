build: 
  make -C build/ -j 12

setup-build:
  mkdir -p build
  cmake -DMLIR_DIR=${MLIR_DIR} -DONNX_USE_PROTOBUF_SHARED_LIBS=ON -Bbuild -S.

run-fast-tests: build
 python tests/run.py --fast

run-tests: build
 python tests/run.py --fast

run-basic-mnist: build
  ./
