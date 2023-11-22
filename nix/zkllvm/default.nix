{ clangStdenv, fetchgit, cmake, boost180, spdlog, zlib, zstd, python3, icu70 }:

clangStdenv.mkDerivation {
  name = "zkllvm_with_mlir";
  src = fetchgit {
    url = "https://github.com/NilFoundation/zkllvm";
    rev = "refs/tags/v0.1.9"; # also change hash + version below in cmakeflags
    sha256 = "1639szlda7jdcag90qdzvkkyq02n10h9ljhz0drzw233vz2y7qsq";
    fetchSubmodules = true;
  };
  enableParallelBuilding = true;

  buildInputs = [
    cmake
    (boost180.override { enableShared = false; })
    spdlog
    zlib
    zstd
    icu70
    python3
  ];
  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCIRCUIT_ASSEMBLY_OUTPUT=TRUE"
    "-DLLVM_ENABLE_PROJECTS=clang;mlir"
    "-DLLVM_TARGETS_TO_BUILD=Assigner;X86"
    "-DLLVM_ENABLE_ASSERTIONS=ON"
    "-DLLVM_ENABLE_RTTI=ON"
    "-DLLVM_ENABLE_LIBEDIT=OFF"
    "-DZKLLVM_VERSION=v0.1.9"
  ];
  installPhase = ''
    mkdir -p $out
    cp -r $src/* $out
  '';
}
