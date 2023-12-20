{ clangStdenv, fetchgit, cmake, boost180, zlib, zstd, python3, icu70 }:

clangStdenv.mkDerivation {
  name = "zkllvm_with_mlir";
  src = fetchgit {
    url = "https://github.com/NilFoundation/zkllvm";
    rev = "642f678b027ccd965d2da384301c3217e3530206"; # also change hash + version below in cmakeflags
    sha256 = "1chj928acqpyrsvb7g3yhmhn32vj3d7b2h1a9ga008gpblpfhgyb";
    fetchSubmodules = true;
  };
  enableParallelBuilding = true;

  buildInputs = [
    cmake
    (boost180.override { enableShared = false; })
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
    "-DLLVM_INSTALL_UTILS=ON"
    "-DZKLLVM_VERSION=v0.1.11"
  ];
  postInstall = ''
    rm -r $out/lib/cmake/crypto3_algebra
    rm -r $out/lib/cmake/crypto3_multiprecision
    rm -r $out/lib/cmake/crypto3_zk
  '';
}
