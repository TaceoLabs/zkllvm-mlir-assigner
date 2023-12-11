{ clangStdenv, fetchgit, cmake, boost180, zlib, zstd, python3, icu70 }:

clangStdenv.mkDerivation {
  name = "zkllvm_with_mlir";
  src = fetchgit {
    url = "https://github.com/NilFoundation/zkllvm";
    rev = "47167ae745c198af8c21f1361aed2d15be223301"; # also change hash + version below in cmakeflags
    sha256 = "0d77yasbcpyk21dih3w2y2lsjxkrxcd9vj67z4ypkzc3jlnsggah";
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
    "-DZKLLVM_VERSION=v0.1.10"
  ];
  postInstall = ''
    rm -r $out/lib/cmake/crypto3_algebra
    rm -r $out/lib/cmake/crypto3_multiprecision
    rm -r $out/lib/cmake/crypto3_zk
  '';
}
