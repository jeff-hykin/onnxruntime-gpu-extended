# onnxruntime-gpu-extended

Pre-built ONNX Runtime **CUDA** wheels for platforms that upstream
`onnxruntime-gpu` does **not** ship — primarily **NVIDIA Jetson** (aarch64 +
CUDA / L4T) — published under a single package name so installation is the same
everywhere.

There is no source code in this repo. It downloads official CUDA wheels from
PyPI (`onnxruntime-gpu`) and builds the missing aarch64 Jetson wheels with Nix
(see `flake.nix` + `TARGETS.md`). All wheels are renamed to
`onnxruntime-gpu-extended` and uploaded to PyPI with a unified version, keeping
the internal `onnxruntime` package intact so `import onnxruntime` works
unchanged.

## Install

```sh
pip install onnxruntime-gpu-extended
```

Works as a drop-in replacement — just `import onnxruntime` as usual.

## Status

**Placeholder / name reserved (0.0.1).** The Jetson wheels build (see CI), but
publishing the real wheels is blocked on:

1. **PyPI 100 MB per-file limit.** Upstream `onnxruntime-gpu` wheels are
   ~200 MB and the JetPack 6 (CUDA 12.6) wheels are ~150 MB — both over the
   default limit. A per-project file-size increase request to PyPI is required.
2. **Version alignment.** nixpkgs' `onnxruntime` (the base for the Jetson
   build) is currently 1.22.x while upstream `onnxruntime-gpu` is 1.27.x. The
   republished set should agree on a version.
3. **Jetson wheel portability.** Nix-built wheels carry `/nix/store` rpaths and
   are not yet stock-Jetson portable (repath/bundle against JetPack-provided
   CUDA is a separate step), plus no runtime GPU smoke-test yet.

## Platform Coverage (target)

| | Linux x86_64 (CUDA) | Windows (CUDA) | Linux aarch64 / Jetson (CUDA) |
|---|---|---|---|
| Python 3.10 | upstream | upstream | nix (JetPack 5/6) |
| Python 3.11 | upstream | upstream | nix (JetPack 5/6) |
| Python 3.12 | upstream | upstream | nix (JetPack 5/6) |
| Python 3.13 | upstream | upstream | nix (best-effort) |
| Python 3.14 | upstream | upstream | nix (best-effort) |

## Building & Publishing

- `nix build .#jp6-cp312` (on a native aarch64 runner) — build a Jetson wheel.
  See `TARGETS.md` for the full matrix and `.github/workflows/build-jetson-nix.yml`.
- `run/publish` — download upstream wheels, collect local Nix builds, rename all
  to `onnxruntime-gpu-extended`, and upload to PyPI. `PUBLISH_SKIP_UPLOAD=1` does
  a dry run (build + rename only).
