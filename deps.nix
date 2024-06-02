with import <nixpkgs> {};
let
  hatch-config = ({
    x86_64-linux = {
      system = "x86_64-unknown-linux-gnu";
      sha256 = "01dm83vr9gcdva2hdbvsifvnmfg89gkm0crvavmsd9xvpcqbj828";
    };
    aarch64-linux = {
      system = "aarch64-unknown-linux-gnu";
      sha256 = "1xb56qmy6ksrz1m7h3hxhcjnysspijqyy8yqivzsjf7jnq72k5xy";
    };
    x86_64-darwin = {
      system = "x86_64-apple-darwin";
      sha256 = "1c6rpp5xq3d5jmx47mjsp4pmcx96c1199smjvkjkp1c5x7wklanx";
    };
    aarch64-darwin = {
      system = "aarch64-apple-darwin";
      sha256 = "0wkpl230mxf9ysbcv7q84bm66rk4lcgscf039fhgp0h8avm07ffx";
    };
  })."${builtins.currentSystem}";
in {
  hatch = stdenv.mkDerivation {
    name = "hatch";
    src = (builtins.fetchurl {
      url = "https://github.com/pypa/hatch/releases/download/hatch-v1.10.0/hatch-1.10.0-${hatch-config.system}.tar.gz";
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
