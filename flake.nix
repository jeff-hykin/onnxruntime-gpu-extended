{
  description = "Build onnxruntime-gpu (CUDA) wheels for platforms upstream doesn't ship — primarily NVIDIA Jetson (aarch64 + CUDA), cross-compiled from x86_64.";

  # ===========================================================================
  # STRATEGY (staged)
  #
  # Upstream `onnxruntime-gpu` on PyPI ships CUDA wheels for linux x86_64 +
  # windows only. The gap we fill: NVIDIA Jetson (aarch64 + CUDA / L4T). That
  # build is the whole point of this repo.
  #
  # Stage 1 (here, WIP): build an onnxruntime-gpu wheel WITH the CUDA execution
  #   provider natively on an x86_64-linux + NVIDIA host. This proves the nix
  #   build of onnxruntime+CUDA works and produces a redistributable wheel.
  #   NOTE: cannot be built on macOS (nixpkgs CUDA is linux-only). Build on a
  #   real x86_64 CUDA box.
  #
  # Stage 2 (TODO): cross-compile the same wheel for Jetson (aarch64 + CUDA)
  #   from an x86_64 host. Plan: use jetpack-nixos (anduril) which provides the
  #   L4T BSP — CUDA, cuDNN, TensorRT for aarch64 Jetson — and is designed for
  #   x86_64 -> aarch64 cross builds. onnxruntime's CUDA EP compiles device code
  #   to cubin/PTX for the target SM (Orin sm_87, Xavier sm_72, Nano sm_53)
  #   without needing a GPU present at build time, so cross-compiling is viable.
  #
  # Key known hard problem: onnxruntime's build pulls many deps via CMake
  # FetchContent at configure time (abseil, protobuf, flatbuffers, onnx, eigen,
  # re2, ...). In the nix sandbox (no network) those must be pre-vendored.
  # nixpkgs' own `onnxruntime` derivation already solves this, so Stage 1 builds
  # ON TOP of nixpkgs' onnxruntime rather than from raw source.
  # ===========================================================================

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # L4T / JetPack (CUDA, cuDNN, TensorRT) for Jetson + x86_64->aarch64 cross.
    # Used in Stage 2.
    jetpack-nixos.url = "github:anduril/jetpack-nixos";
  };

  outputs = { self, nixpkgs, flake-utils, jetpack-nixos, ... }:
    let
      # onnxruntime release to track/build. The -extended version will follow
      # upstream onnxruntime-gpu (single source of truth, à la gtsam_extended's
      # run/upstream_version.py — to be added once the build itself works).
      onnxruntimeVersion = "1.20.1"; # TODO: confirm/pin desired upstream release

      # CUDA requires unfree + the cudaSupport config flag.
      pkgsFor = system: import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
    in
    # Stage 1 only targets x86_64-linux. (aarch64 Jetson handled in Stage 2.)
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = pkgsFor system;
        python = pkgs.python311;
        cuda = pkgs.cudaPackages;
      in {
        # ---------------------------------------------------------------------
        # Stage 1 draft: x86_64-linux + CUDA onnxruntime, wheel extraction TODO.
        #
        # UNTESTED — authored on a Mac that cannot evaluate CUDA derivations.
        # First real iteration must happen on an x86_64 CUDA host:
        #   nix build .#onnxruntime-gpu-wheel
        # ---------------------------------------------------------------------
        packages.onnxruntime-gpu-wheel =
          (python.pkgs.onnxruntime.override {
            cudaSupport = true;
          }).overrideAttrs (old: {
            # TODO(on-machine): onnxruntime's python build leaves a setup.py in
            # the cmake build tree. Add a postBuild/installPhase that runs
            # `python setup.py bdist_wheel` there and copies the .whl into $out,
            # mirroring gtsam_extended's flake (setup.py bdist_wheel -> repack).
            # Then decide CUDA-lib bundling: Jetson provides CUDA via JetPack, so
            # the wheel likely should NOT bundle libcudart/cudnn/etc.
          });

        # Toolchain for driving/iterating the build by hand on a CUDA host.
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.cmake
            pkgs.ninja
            python
            python.pkgs.numpy
            python.pkgs.setuptools
            python.pkgs.wheel
            python.pkgs.packaging
            cuda.cuda_nvcc
            cuda.cudatoolkit
            cuda.cudnn
          ];
          shellHook = ''
            echo "onnxruntime-gpu-extended dev shell (onnxruntime ${onnxruntimeVersion}, CUDA)"
            echo "  Stage 1: nix build .#onnxruntime-gpu-wheel  (x86_64 + CUDA host only)"
          '';
        };
      });
}
