#! /usr/bin/env nix
#! nix --extra-experimental-features ``flakes nix-command`` shell
#! nix --override-input nixpkgs ``github:NixOS/nixpkgs/release-23.11``
#! nix --impure --file deps.nix hatch bash
#! nix --command bash

echo 'Enter Dev Shell'
bash
echo 'Exit Dev Shell'
