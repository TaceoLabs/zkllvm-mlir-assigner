# zkLLVM nix derivation

This contains a nix derivation for the specific version of zkLLVM we target for compatibility.

To build it standalone:

```bash
nix-build --expr 'with import <nixpkgs> {}; callPackage ./default.nix {}' --cores 16
```

To update it use `nix-prefetch-git` to help get the required hashes:

```bash
nix-prefetch-git --url "https://github.com/NilFoundation/zkllvm" --rev "c1bc3aa86905a9e8548514779a51aee27187b90d" --fetch-submodules
```
