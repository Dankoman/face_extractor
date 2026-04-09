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
    pkgs.onnxruntime
    pkgs.opencv
    pkgs.swi-prolog
    pkgs.tmux
    pkgs.byobu
    pkgs.chafa
    pkgs.patchelf
  ];

  shellHook = ''
    echo "✅ Nix-miljö laddad (med nix-ld och swi-prolog support)."

    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
      pkgs.zlib
      pkgs.glib
      pkgs.onnxruntime
      pkgs.opencv
      pkgs.swi-prolog
      pkgs.dbus
      pkgs.atk
      pkgs.pango
      pkgs.gtk3
    ]}:''${LD_LIBRARY_PATH:-}"

    # nix-ld support
    export NIX_LD="${pkgs.stdenv.cc.libc}/lib/ld-linux-x86-64.so.2"
    export NIX_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

    export STASHDB_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOiJlYjAzMzRkNi03NTQ4LTRhYjAtYjExMC0xOGEyZmI2Y2YwMDQiLCJzdWIiOiJBUElLZXkiLCJpYXQiOjE2NjM0NDg1MjZ9.ckvz_oNpI_HSCGSSOa1xI2mprnoDCBl7EuoBAXcK6Us"
    export TPDB_API_KEY="uBNsAhIe9evycv8q10x1wkRlEzFtFmYUTimvll0416e2b867"
    export PMVSTASH_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOiIzMDY5MmNmNS0yMjlmLTRiY2YtOTlkNi0xZWY5NDRkZjlhZjIiLCJzdWIiOiJBUElLZXkiLCJpYXQiOjE2OTU3MTk4MzZ9.7xFdhw7sRUfeOOUL3evpnhX0j5NmP4SAj4MOO7WzhN4"
    export FANSDB_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOiI5YWE1NjNmOS04ZjkyLTRlOTgtYTg3MC0wNWYzNDIzZmQwZWYiLCJzdWIiOiJBUElLZXkiLCJpYXQiOjE3MDgwODQ2MjB9.dyrsw96W41h3M2VFX37rpxcJUkpHg21iF2KGFmOSWsU"
    export STRICT_NAME_MATCH=1
    export FACE_EXTRACTOR_FEMALE_THRESHOLD="0.3"


    if [ ! -d .venv ]; then
      python3 -m venv .venv
    fi
    source .venv/bin/activate

    if [ -z "$SKIP_PIP" ]; then
      pip install --upgrade pip
      pip install --no-cache-dir onnxruntime insightface opencv-python-headless==4.12.0.88 numpy flask flask-cors rich pyswip
    fi
    
    if [ -z "$SKIP_API" ]; then
      python api_endpoint.py
    fi
  '';
}
