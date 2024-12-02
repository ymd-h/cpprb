with import <nixpkgs> {};
let
  hatch-config = ({
    x86_64-linux = {
      system = "x86_64-unknown-linux-gnu";
      sha256 = "0i7vf48350mcwz6vcd46awsblm36a6ih7vrfjadppij0a6047apd";
    };
    aarch64-linux = {
      system = "aarch64-unknown-linux-gnu";
      sha256 = "0lxnkjlfff81zv34d0l5cjybgyxa5lz0p8lm1gzl2mf8alwgbv7i";
    };
    x86_64-darwin = {
      system = "x86_64-apple-darwin";
      sha256 = "1nl036mqvbq38imna22ign2wb1b255qn3vsq3vr4n993fqq6jlhx";
    };
    aarch64-darwin = {
      system = "aarch64-apple-darwin";
      sha256 = "0y5hscgwfcrl2nk92mzbxqs882qmwz0hjlkay0hxav2yk1gs27mf";
    };
  })."${builtins.currentSystem}";
in {
  hatch = stdenv.mkDerivation {
    name = "hatch";
    src = (builtins.fetchurl {
      url = "https://github.com/pypa/hatch/releases/download/hatch-v1.13.0/hatch-${hatch-config.system}.tar.gz";
      sha256 = hatch-config.sha256;
    });
    phases = ["installPhase" "patchPhase"];
    installPhase = ''
      mkdir -p $out/bin
      tar -xzf $src -O > $out/bin/hatch
      chmod +x $out/bin/hatch
    '';
  };
  bash = bashInteractive;
}
