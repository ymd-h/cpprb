#! /usr/bin/env nix
#! nix --extra-experimental-features ``flakes nix-command`` shell
#! nix -I nixpkgs=https://github.com/NixOS/nixpkgs/archive/branch-off-24.11.tar.gz
#! nix --impure nixpkgs#actionlint nixpkgs#uv
#! nix --command bash

echo 'Enter Dev Shell'
bash
echo 'Exit Dev Shell'
