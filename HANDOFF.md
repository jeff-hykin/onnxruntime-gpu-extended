# onnxruntime-gpu-extended — Build & Cross-Compile Handoff

A task brief for an agent picking up this repo. Goal: publish `onnxruntime-gpu`
CUDA wheels for the platform upstream doesn't ship — **NVIDIA Jetson (aarch64 +
CUDA / L4T)** — cross-compiled from x86_64, under a single unified PyPI name,
following the same pattern as `gtsam-extended`.

## Links

- This repo (local only, **no GitHub remote yet** — do not create one without
  asking Jeff): `~/repos/onnxruntime-gpu-extended`
- Upstream onnxruntime source: https://github.com/microsoft/onnxruntime
- Upstream PyPI package we mirror/extend: `onnxruntime-gpu`
- jetpack-nixos (L4T/JetPack for Nix, x86_64→aarch64 Jetson cross):
  https://github.com/anduril/jetpack-nixos
- Pattern to copy (the working sibling repo): https://github.com/jeff-hykin/gtsam-extended

## The core constraints (read first — they shape everything)

1. **No GPU is needed to *compile* CUDA.** `nvcc` emits PTX/cubin for a target
   SM arch (Orin `sm_87`, Xavier `sm_72`, Nano `sm_53`) with no GPU present. A
   GPU is only needed to *runtime-test* that a built wheel actually loads
   `CUDAExecutionProvider`.

2. **You cannot build on macOS.** `nvcc` and its internals (`ptxas`, `cicc`,
   `fatbinary`, `nvlink`) are closed-source prebuilt NVIDIA blobs, shipped only
   for `linux-x86_64`, `linux-aarch64` (Jetson/sbsa), and `windows-x86_64`.
   There is no macOS build, so nixpkgs `cudaPackages` is Linux-only. The
   compiler that must *run on the host* doesn't exist for darwin. Cross-
   compiling retargets the *output*, not the *host the compiler runs on*.

3. **The x86_64 → Jetson-aarch64 cross DOES work** — NVIDIA ships that cross
   toolchain (runs on a Linux x86_64 host, targets `sm_87`). The host just has
   to be Linux. This is Stage 2.

4. **Therefore the whole build runs on Linux x86_64 — no GPU, no Jetson, no
   special hardware.** A free **GitHub Actions `ubuntu` x86_64 runner** is
   enough for both stages. This is the key unblock: the build does NOT depend on
   any reachable CUDA box. (Runtime verification is the only GPU-dependent step,
   deferred to a Jetson when one is reachable.)

5. **onnxruntime's CMake build pulls deps via FetchContent at configure time**
   (abseil, protobuf, flatbuffers, onnx, eigen, re2, …). The nix sandbox has no
   network, so these must be pre-vendored. **nixpkgs' own `onnxruntime`
   derivation already solves this**, so build ON TOP of nixpkgs' onnxruntime
   (override `cudaSupport = true`) rather than from raw upstream source.

## Current state of the repo

- `flake.nix` — scaffolded, **UNTESTED** (authored on a Mac that can't eval
  CUDA). Inputs: nixpkgs (unstable) + flake-utils + jetpack-nixos. Config
  `allowUnfree = true; cudaSupport = true`. Stage-1 draft package
  `onnxruntime-gpu-wheel` = `python311.pkgs.onnxruntime.override { cudaSupport
  = true; }` with a `overrideAttrs` stub — **wheel extraction is a TODO**.
  devShell has cmake/ninja/python/cuda toolchain. `onnxruntimeVersion =
  "1.20.1"` is a placeholder.
- `flake.lock` — locked (resolves on the Mac; CUDA derivations just can't be
  *built* there).
- `.gitignore` — result, result-*, wheels_input/, renamed_wheels/, __pycache__.
- One commit (`9dc775e`). Branch `main`. No remote.

## Plan

### Stage 1 — build the x86_64-linux CUDA wheel (no GPU needed)

Do this first to prove the Nix CUDA build works and produces a redistributable
wheel.

1. On a Linux x86_64 host (a GitHub `ubuntu` runner is fine), install Nix
   (DeterminateSystems installer) and run `nix build .#onnxruntime-gpu-wheel`.
2. Finish the wheel-extraction TODO in `flake.nix`: onnxruntime's python build
   leaves a `setup.py` in the cmake build tree. Add a `postBuild`/`installPhase`
   that runs `python setup.py bdist_wheel` there and copies the `.whl` into
   `$out` — mirror how `gtsam-extended`'s flake does `setup.py bdist_wheel` →
   repack.
3. Decide CUDA-lib bundling. Jetson provides CUDA via JetPack at runtime, so the
   Jetson wheel likely should **NOT** bundle `libcudart`/`cudnn`/etc. For an
   x86_64 wheel you may want manylinux-style bundling (auditwheel) — but since
   the real target is Jetson, keep Stage-1 minimal and don't over-invest in the
   x86_64 wheel's portability.
4. Confirm the wheel imports and (on a GPU box, later) that
   `onnxruntime.get_available_providers()` lists `CUDAExecutionProvider`.

### Stage 2 — cross-compile for Jetson (aarch64 + CUDA / L4T)

This is the actual deliverable.

1. Add jetpack-nixos as the source of the aarch64 CUDA/cuDNN/TensorRT stack and
   set up an `x86_64-linux` → `aarch64-linux` cross build (jetpack-nixos is
   designed for exactly this). See its README/examples for the cross
   `pkgsCross` wiring and L4T package set.
2. Build onnxruntime's CUDA EP targeting the Jetson SM (`sm_87` for Orin — pick
   per the target board). Device code compiles to cubin/PTX without a GPU.
3. Produce an aarch64 manylinux/L4T-compatible wheel. Do NOT bundle the CUDA
   runtime — link against the JetPack-provided libs the device already has.
4. Pick the JetPack/L4T version to match the target Jetson's flashed BSP
   (confirm with Jeff which board + JetPack version).

### Stage 3 — publish pipeline + auto-update loop (copy gtsam-extended)

Once both wheels build, replicate `gtsam-extended`'s republishing machinery:

1. `upstream_version.py`-style single-source-of-truth that tracks the latest
   `onnxruntime-gpu` release on PyPI and derives the `-extended` version.
2. `download_wheels.py` to pull upstream's existing wheels (linux x86_64 +
   windows) from PyPI.
3. `rename_wheel.py` to rename everything (upstream wheels + our Jetson wheel)
   to the unified `onnxruntime-gpu-extended` name + normalized version, rewriting
   RECORD.
4. `publish.yml` GitHub workflow (ubuntu x86_64 runner): build our Jetson wheel
   (Nix), download the rest, rename, `twine check`, `twine upload
   --skip-existing`. Needs a `PYPI_TOKEN` secret — **Jeff must add this himself;
   never move his PyPI token off his machine into CI without explicit per-task
   approval.**
5. `auto-update.yml` daily cron that compares upstream vs published and triggers
   publish on a new release.

## Gotchas / open questions for Jeff

- Which Jetson board(s) + JetPack/L4T version are the target? (Determines SM
  arch and the jetpack-nixos L4T pin.)
- Confirm the upstream `onnxruntime-gpu` version to track (placeholder is
  1.20.1).
- Runtime verification of `CUDAExecutionProvider` needs a real Jetson/GPU — none
  were reachable over SSH as of 2026-06-27 (Tailscale/VPN down). Build + publish
  do not need it; only the final smoke-test does.
- nixpkgs' `onnxruntime` may lag the latest upstream onnxruntime release; check
  the version it pins and whether it matters for the target.

## Standing rules (Jeff's — do not violate)

- Never push or create remotes/PRs without explicit per-task approval (previous
  approvals don't carry forward).
- Never move/copy/upload Jeff's credentials (PyPI token, `.pypirc`, `.env`) into
  CI or any external system without asking first. Reading locally to run a
  command is fine.
- Never enable PyPI Trusted Publishing / OIDC without asking.
- Coding style: 4-space indent, no semicolons, `{}` on all control flow, prefer
  `flake.nix` for deps.
