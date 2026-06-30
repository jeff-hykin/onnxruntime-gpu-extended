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
    # jetpack-nixos's CUDA overlay (cudaPackages_<ver> manifest overrides) is written
    # against the nixpkgs release it pins (nixos-25.11). Feeding it a different nixpkgs
    # (e.g. unstable) breaks with "manifest version missing", so the Jetson packages are
    # built from a matching nixos-25.11 that jetpack-nixos `follows`.
    nixpkgs-jetpack.url = "github:NixOS/nixpkgs/nixos-25.11";
    # L4T / JetPack (CUDA, cuDNN, TensorRT) for Jetson. Provides nvidia-jetpack5/6/7
    # and the matching cudaPackages_<ver> sets via its overlay.
    jetpack-nixos.url = "github:anduril/jetpack-nixos";
    jetpack-nixos.inputs.nixpkgs.follows = "nixpkgs-jetpack";
  };

  outputs = { self, nixpkgs, nixpkgs-jetpack, flake-utils, jetpack-nixos, ... }:
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

      # Python package set for a wheel build. For cp310 we pin numpy 2.2.6: nixos-25.11
      # ships numpy 2.3.x which dropped Python 3.10 (so python310Packages.numpy throws
      # "not supported for interpreter"), and 2.2.6 is the last series supporting 3.10.
      # Everything else uses the stock set.
      pyPackagesFor = ortPkgs: ver:
        let base = ortPkgs.${"python${ver}Packages"};
        in if ver != "310" then base
        else base.overrideScope (pyself: pysuper: {
          numpy = pysuper.numpy.overridePythonAttrs (old: {
            version = "2.2.6";
            src = pysuper.fetchPypi {
              pname = "numpy";
              version = "2.2.6";
              hash = "sha256-4pVU4r71SpCqXMB9ps6VWsy4PyGrXeAaYshHiJeyZP0=";
            };
            # The version-specific guard and 2.3.x patches don't apply to 2.2.6.
            disabled = false;
            patches = [ ];
          });
        });

      # Build the wheel (`dist` output) for a given configured pkgs instance and
      # python version. `ortPkgs` is the package set whose `onnxruntime` carries
      # the desired CUDA set (e.g. pkgs.cudaPackages_12_6.pkgs for JetPack 6).
      mkWheel = ortPkgs: ver:
        let
          ort = ortPkgs.onnxruntime.override {
            python3Packages = pyPackagesFor ortPkgs ver;
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

        # aarch64 Jetson: nixpkgs + jetpack-nixos overlay. We select the JetPack CUDA
        # set per target via cudaPackages_<ver>.pkgs (a nixpkgs instance whose default
        # cudaPackages is that set, so `onnxruntime` links the right CUDA).
        #
        pkgsJetson = caps: import nixpkgs-jetpack {
          inherit system;
          config = cudaConfig caps;
          overlays = [ jetpack-nixos.overlays.default ];
        };

        # JetPack target -> CUDA package-set attr + SM caps. Orin=sm_87, Xavier=sm_72, Thor=cc11.0.
        jetsonTargets = {
          jp6 = { cudaAttr = "cudaPackages_12_6"; caps = [ "8.7" ]; };       # CUDA 12.6
          jp5 = { cudaAttr = "cudaPackages_11"; caps = [ "7.2" "8.7" ]; };   # CUDA 11.4
          jp7 = { cudaAttr = "cudaPackages_13_0"; caps = [ "11.0" ]; };      # CUDA 13.0
        };

        # The onnxruntime package set for a JetPack target: take that JetPack's
        # cudaPackages.pkgs. Returns null if that CUDA set isn't present on this
        # nixpkgs (e.g. no CUDA 13 set for jp7), so the combo is skipped instead of
        # breaking the whole attrset eval. (`ver` is unused here — the per-python
        # numpy pin lives in mkWheel/pyPackagesFor.)
        ortPkgsFor = jpTag: ver:
          let
            t = jetsonTargets.${jpTag};
            base = pkgsJetson t.caps;
          in
          if base ? ${t.cudaAttr} then base.${t.cudaAttr}.pkgs else null;

        x86Wheels = wheelsFor pkgsX86 "x86";
        jetsonWheels = builtins.listToAttrs (builtins.concatMap
          (jpTag: builtins.concatMap
            (ver:
              let ortPkgs = ortPkgsFor jpTag ver;
              in if ortPkgs != null && hasPy ortPkgs ver
                 then [{ name = "${jpTag}-cp${ver}"; value = mkWheel ortPkgs ver; }]
                 else [ ])
            pythonVersions)
          (builtins.attrNames jetsonTargets));

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
