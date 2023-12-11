{
  description = "C++ Development Shell";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:

      let
        pkgs = import nixpkgs {
          inherit system;
        };
        zkllvm_with_mlir = (pkgs.callPackage ./nix/zkllvm { });
        python-packages = ps: with ps; [ numpy onnx ];
      in
      with pkgs;
      {
        devShells.default = mkShell.override { stdenv = clangStdenv; }
          {
            buildInputs = [
              (pkgs.python3.withPackages python-packages)
              pkgs.clang-tools
              pkgs.cmake
              pkgs.ninja
              pkgs.git
              pkgs.ripgrep
              pkgs.protobuf3_20
              pkgs.zlib
              pkgs.zstd
              pkgs.libxml2
              pkgs.icu70
              pkgs.just
              zkllvm_with_mlir
              # (pkgs.callPackage ./nix/zkllvm/default.nix { inherit pkgs; })
              (pkgs.boost180.override { enableShared = false; })
            ];
            shellHook = ''
              export MLIR_DIR=${zkllvm_with_mlir}/lib/cmake/mlir
              echo Using MLIR in $MLIR_DIR
            '';
          };
      }
    );
}
