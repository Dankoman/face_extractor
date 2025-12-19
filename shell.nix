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
    pkgs.tmux
    pkgs.byobu
  ];

  shellHook = ''
    echo "✅ Nix-miljö laddad."

    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
    export STASH_URL="http://192.168.0.50:9999"
    export STASHDB_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOiJlYjAzMzRkNi03NTQ4LTRhYjAtYjExMC0xOGEyZmI2Y2YwMDQiLCJzdWIiOiJBUElLZXkiLCJpYXQiOjE2NjM0NDg1MjZ9.ckvz_oNpI_HSCGSSOa1xI2mprnoDCBl7EuoBAXcK6Us"
    export STRICT_NAME_MATCH=1
    export FACE_EXTRACTOR_FEMALE_THRESHOLD="0.3"


    if [ ! -d .venv ]; then
      python3 -m venv .venv
    fi
    source .venv/bin/activate

    if [ -z "$SKIP_PIP" ]; then
      pip install --upgrade pip
      pip install --no-cache-dir onnxruntime insightface opencv-python-headless==4.12.0.88 numpy flask flask-cors
    fi
    
    if [ -z "$SKIP_API" ]; then
      python api_endpoint.py
    fi
  '';
}
