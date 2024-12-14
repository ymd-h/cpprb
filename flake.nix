{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      lib = nixpkgs.lib;
      export = "(progn (require 'ox) (require 'ox-hugo) (setq org-src-preserve-indentation t) (org-hugo-export-wim-to-md :all-subtrees nil t))";
    in
      {
        packages = rec {
          readme = pkgs.stdenv.mkDerivation {
            name = "cpprb-readme";

            src = lib.fileset.toSource {
              root = ./.;
              fileset = lib.fileset.unions [
                ./README.org
                ./LICENSE
              ];
            };

            nativeBuildInputs = [
              pkgs.emacs
            ];

            buildPhase = ''
              emacs --batch README.org --eval "(org-md-export-to-markdown)"
            '';

            installPhase = ''
              mkdir -p $out
              cp ./README.md $out/README.md
            '';
          };

          python-311 = pkgs.callPackage ./cpprb.nix {
            inherit pkgs;
            inherit lib;
            ps = pkgs.python311Packages;
            readme = readme;
          };

          python-312 = pkgs.callPackage ./cpprb.nix {
            inherit pkgs;
            inherit lib;
            ps = pkgs.python312Packages;
            readme = readme;
          };

          site = pkgs.stdenv.mkDerivation {
            name = "cpprb-site";

            src = lib.fileset.toSource {
              root = ./.;
              fileset = lib.fileset.unions [
                ./README.org
                ./CHANGELOG.org
                ./LICENSE
                ./site
                ./sphinx
                ./example
              ];
            };

            nativeBuildInputs = [
              pkgs.hugo
              (pkgs.emacs.pkgs.withPackages (epkgs: (with epkgs.melpaStablePackages; [
                ox-hugo
              ])))
              (pkgs.python312.withPackages (ps: (with ps; [
                numpy
                sphinx
                sphinx-rtd-theme
                sphinx-automodapi
                python-312
              ])))
            ];

            buildPhase = ''
              emacs --batch README.org --eval "${export}"
              emacs --batch CHANGELOG.org --eval "${export}"
              cd site
              emacs --batch site.org --eval "${export}"
              hugo -c content --logLevel info
              cd ../

              mkdir -p public/api
              sphinx-build -b html sphinx public/api
            '';

            installPhase = ''
              mkdir -p $out/public
              cp -r public $out/
            '';
          };
        };
      }
  );
}
