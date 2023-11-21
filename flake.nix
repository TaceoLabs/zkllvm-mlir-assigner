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
        python-packages = ps: with ps; [ numpy onnx ];
      in
      with pkgs;
      {
        devShells.default = mkShell.override { stdenv = clangStdenv; }
          {
            buildInputs = [
              (pkgs.python310.withPackages python-packages)
              pkgs.clang-tools
              pkgs.cmake
              pkgs.ninja
              pkgs.git
              pkgs.ripgrep
              pkgs.protobuf3_20
              pkgs.spdlog
              pkgs.zstd
              (pkgs.boost180.override { enableShared = false; })
            ];
          };
      }
    );
}
