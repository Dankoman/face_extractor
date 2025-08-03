{ pkgs ? import <nixpkgs> {} }:

let
  py = pkgs.python311;
  pyPkgs = py.withPackages (ps: with ps; [
    pip setuptools wheel
    numpy scipy scikit-learn tqdm pillow
  ]);
in
pkgs.mkShell {
  name = "arcface-env";

  buildInputs = [
    pyPkgs
    pkgs.cmake pkgs.pkg-config pkgs.gcc pkgs.gfortran pkgs.openblas
    pkgs.boost pkgs.eigen
    pkgs.curl pkgs.openssl pkgs.git pkgs.zlib
    pkgs.dejavu_fonts
  ];

  shellHook = ''
    echo "✅ Nix-miljö laddad."

    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

    if [ ! -d .venv ]; then
      python3 -m venv .venv
    fi
    source .venv/bin/activate

    pip install --upgrade pip
    pip install --no-cache-dir onnxruntime insightface opencv-python-headless==4.12.0.88 numpy flask

    echo "✅ Virtuell miljö aktiv. Kör: python face_arc_pipeline.py --mode both"
  '';
}
