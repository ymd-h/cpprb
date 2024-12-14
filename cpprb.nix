{
  pkgs,
  lib,
  ps,
  readme,
}: ps.buildPythonPackage {
  pname = "cpprb";

  version = "11.0.0";

  doCheck = false;

  pyproject = true;

  build-system = [
    ps.setuptools
    ps.wheel
    ps.cython
    ps.numpy_2
  ];

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./pyproject.toml
      ./setup.py
      ./LICENSE
      ./MANIFEST.in
      ./src
    ];
  };

  patchPhase = ''
    cp ${readme}/README.md ./README.md
  '';
}
