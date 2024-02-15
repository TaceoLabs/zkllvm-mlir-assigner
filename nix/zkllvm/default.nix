{ clangStdenv, fetchgit, cmake, git, boost180, zlib, zstd, python3, icu70 }:

clangStdenv.mkDerivation {
  name = "zkllvm_with_mlir";
  src = fetchgit {
    url = "https://github.com/NilFoundation/zkllvm";
    rev = "fcf5cc56716c9ac2011ec7b91db45a70224cef93"; # also change hash + version below in cmakeflags
    sha256 = "1iv8ijs4cxnflsiyf0hb6w9cyfdcx835izy5kbvlpj56pn5ymwpq";
    fetchSubmodules = true;
  };
  enableParallelBuilding = true;

  patches = [
    # This patch removes the /etc/ld.conf.d installation which we do not want in nix
    ./patches/0001-patch-do-not-install-linker-config-for-zkllvm-nix.patch
  ];

  buildInputs = [
    cmake
    git
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
    "-DZKLLVM_VERSION=v0.1.16" # WARN: this is not really the exact commit for this version
  ];
  # We need to clean up some files that are invalid for newer cmake versions.
  postInstall = ''
    rm -r $out/lib/cmake/crypto3_algebra
    rm -r $out/lib/cmake/crypto3_multiprecision
    rm -r $out/lib/cmake/crypto3_zk
  '';
}
