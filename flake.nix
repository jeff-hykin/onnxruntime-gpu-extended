{
  description = "Build onnxruntime CUDA wheels for platforms upstream onnxruntime-gpu doesn't ship — primarily NVIDIA Jetson (aarch64 + CUDA / L4T).";

  # ===========================================================================
  # HOW THIS WORKS (read TARGETS.md for the full matrix + reasoning)
  #
  # We build ON TOP OF nixpkgs' own `onnxruntime` derivation (currently 1.26.0).
  # That derivation already:
  #   - pre-vendors every CMake FetchContent dep (abseil, protobuf, onnx, ...),
  #   - builds the python wheel itself (`setup.py bdist_wheel` in postBuild),
  #     leaving the `.whl` in its `dist` output.
  # So a "wheel" target here is just `<onnxruntime>.dist`, with the right python
  # and the right CUDA package set selected.
  #
  # x86_64 (Stage 1, VALIDATION only — upstream already ships x86_64 wheels):
  #   built natively here; runtime-verifiable on a local GPU.
  #
  # Jetson aarch64 (the actual deliverable):
  #   CANNOT be cross-compiled from x86_64 — jetpack-nixos disables CUDA in cross
  #   builds ("not supported upstream") and nixpkgs CUDA does not cross. These
  #   targets therefore live under packages.aarch64-linux.* and must be built on
  #   a NATIVE aarch64 Linux host (a reachable Orin, or a GitHub ubuntu-24.04-arm
  #   runner — no GPU needed to COMPILE). jetpack-nixos provides the L4T CUDA via
  #   its overlay; we pick the JetPack CUDA set with cudaPackages_<ver>.pkgs.
  #
  # NOTE on portability: a nix-built wheel has /nix/store rpaths and is NOT yet a
  # stock-Jetson-portable wheel. Repathing/bundling for JetPack-provided CUDA is a
  # Stage-3 (publish) concern, tracked separately.
  # ===========================================================================

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # L4T / JetPack (CUDA, cuDNN, TensorRT) for Jetson. Provides nvidia-jetpack5/6/7
    # and the matching cudaPackages_<ver> sets via its overlay.
    jetpack-nixos.url = "github:anduril/jetpack-nixos";
  };

  outputs = { self, nixpkgs, flake-utils, jetpack-nixos, ... }:
    let
      # Python versions we attempt wheels for. 3.14 is best-effort (very new;
      # onnxruntime may not build against it yet). Keyed by cpXY tag.
      pythonVersions = [ "310" "311" "312" "313" "314" ];

      # Pick a python interpreter attr (pythonXYZ) from a pkgs instance, skipping
      # versions that instance doesn't provide.
      pyAttr = ver: "python${ver}";
      hasPy = pkgs: ver: pkgs ? ${pyAttr ver};

      # Base nixpkgs config for any CUDA build. caps = list of cudaCapabilities
      # (SM targets), e.g. [ "8.7" ] for Orin.
      cudaConfig = caps: {
        allowUnfree = true;
        cudaSupport = true;
        cudaCapabilities = caps;
      };

      # Build the wheel (`dist` output) for a given configured pkgs instance and
      # python version. `ortPkgs` is the package set whose `onnxruntime` carries
      # the desired CUDA set (e.g. pkgs.cudaPackages_12_6.pkgs for JetPack 6).
      mkWheel = ortPkgs: ver:
        let
          py = ortPkgs.${pyAttr ver};
          ort = ortPkgs.onnxruntime.override {
            python3Packages = ortPkgs.${"python${ver}Packages"};
          };
        in
        # The wheel lives in the `dist` output of the onnxruntime build.
        ort.dist;

      # Turn a (configured pkgs instance, target tag) into an attrset of
      # { "<tag>-cp<ver>" = wheel; } for every python version that instance has.
      wheelsFor = ortPkgs: tag:
        builtins.listToAttrs (builtins.concatMap
          (ver:
            if hasPy ortPkgs ver
            then [{ name = "${tag}-cp${ver}"; value = mkWheel ortPkgs ver; }]
            else [ ])
          pythonVersions);
    in
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        # x86_64 validation: build for sm_90 + PTX. nixpkgs CUDA (<12.8) has no
        # sm_120 SASS for Blackwell (RTX 50xx); compute_90 PTX JITs forward onto
        # it. Single arch keeps the build fast.
        pkgsX86 = import nixpkgs {
          inherit system;
          config = cudaConfig [ "9.0" ];
        };

        # aarch64 Jetson: nixpkgs + jetpack-nixos overlay. We select the JetPack
        # CUDA set per target via cudaPackages_<ver>.pkgs (a nixpkgs instance whose
        # default cudaPackages is that set, so `onnxruntime` links the right CUDA).
        pkgsJetson = caps: import nixpkgs {
          inherit system;
          config = cudaConfig caps;
          overlays = [ jetpack-nixos.overlays.default ];
        };

        # Orin = sm_87, Xavier = sm_72, Thor = cc 11.0.
        jp6 = pkgsJetson [ "8.7" ];          # JetPack 6, CUDA 12.6
        jp5 = pkgsJetson [ "7.2" "8.7" ];    # JetPack 5, CUDA 11.4 (Orin+Xavier)
        jp7 = pkgsJetson [ "11.0" ];         # JetPack 7, CUDA 13.0 (Thor)

        x86Wheels = wheelsFor pkgsX86 "x86";
        jetsonWheels =
          (wheelsFor jp6.cudaPackages_12_6.pkgs "jp6")
          // (wheelsFor jp5.cudaPackages_11.pkgs "jp5")
          // (wheelsFor jp7.cudaPackages_13_0.pkgs "jp7");

        packages =
          if system == "aarch64-linux"
          then jetsonWheels
          else x86Wheels // { default = x86Wheels."x86-cp312" or null; };

        devPkgs = pkgsX86;
      in
      {
        inherit packages;

        devShells.default = devPkgs.mkShell {
          packages = [
            devPkgs.cmake
            devPkgs.ninja
            devPkgs.python312
            devPkgs.python312Packages.numpy
            devPkgs.python312Packages.setuptools
            devPkgs.python312Packages.wheel
            devPkgs.python312Packages.packaging
          ];
          shellHook = ''
            echo "onnxruntime-gpu-extended dev shell"
            echo "  x86_64 wheels (validation):  nix build .#x86-cp312"
            echo "  Jetson wheels (native aarch64 builder only):"
            echo "    nix build .#jp6-cp312   # Orin sm_87, CUDA 12.6"
            echo "    nix build .#jp5-cp312   # Orin+Xavier, CUDA 11.4 (needs older ort)"
            echo "    nix build .#jp7-cp312   # Thor, CUDA 13.0"
          '';
        };
      });
}
