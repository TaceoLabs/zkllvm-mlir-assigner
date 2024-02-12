# zkLLVM nix derivation

This contains a nix derivation for the specific version of zkLLVM we target for compatibility.

To build it standalone:

```bash
nix-build --expr 'with import <nixpkgs> {}; callPackage ./default.nix {}' --cores 16
```
