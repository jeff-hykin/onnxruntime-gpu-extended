# Build target matrix (prioritized)

Goal: publish CUDA `onnxruntime` wheels for **NVIDIA Jetson** (aarch64 + CUDA / L4T)
that upstream `onnxruntime-gpu` does NOT ship, under the unified PyPI name
`onnxruntime-gpu-extended`. Upstream already ships linux-x86_64 + windows CUDA
wheels, so x86_64 here is **validation only** (Stage 3 will rename upstream's x86
wheels rather than rebuild them).

## Platform / toolchain reality (read first)

- **No GPU is needed to compile** CUDA device code (PTX/cubin emitted for a target
  SM). A GPU is only needed to *runtime smoke-test* a finished wheel.
- **Jetson CUDA wheels cannot be cross-compiled from x86_64.** jetpack-nixos sets
  `configureCuda = false` in its cross configs ("cross-compilation isn't currently
  supported for cuda packages upstream"), and nixpkgs CUDA does not cross. => the
  aarch64 wheels must be built on a **native aarch64 Linux host** (a reachable Orin,
  or a GitHub `ubuntu-24.04-arm` runner — neither needs a GPU to compile).
- jetpack-nixos supports JetPack **5 / 6 / 7**. **Jetson Nano / TX2 / TX1 are NOT
  supported** (dropped upstream in JP5) → **sm_53 is unreachable** via this toolchain.

## JetPack → L4T → CUDA → SM mapping

| JetPack | L4T   | CUDA (native)     | Boards (SM)                       | onnxruntime that fits |
|---------|-------|-------------------|-----------------------------------|-----------------------|
| 6.x     | r36   | 12.6 (12.4–12.9)  | Orin AGX/NX/Nano (**sm_87**)      | **1.26.0** (nixpkgs) ✓ |
| 5.x     | r35   | 11.4 (→12.2 compat)| Orin (sm_87) + Xavier (**sm_72**) | **1.18.1** (CUDA 11) ✓ |
| 7.x     | r38?  | 13.0              | Thor AGX (**sm_110 / cc 11.0**)   | 1.26.0 (if CUDA13 builds) |

Notes:
- **JP6 is the clean target**: CUDA 12.6 pairs with the onnxruntime 1.26.0 that
  nixpkgs already vendors (all FetchContent deps + a `dist` wheel output).
- **JP5 needs onnxruntime 1.18.1.** 1.18.1 is the last release published to PyPI with
  CUDA 11, and its cmake sets 11.4 as the hard floor (`FATAL_ERROR` below that) —
  exactly what JP5 ships. Flash and memory-efficient attention switch themselves off
  under 11.6, so those kernels are absent from a JP5 wheel.
  Five things differ from the JP6 recipe, all now build args: the base is Ubuntu
  20.04 so deadsnakes has no packages (use `PYTHON_SOURCE=uv`), the host compiler has
  to be `GCC_VERSION=10` (CUDA rejects gcc-12 here, but focal's default gcc-9 lacks
  `-march=armv8.2-a+bf16` and onnxruntime hard-fails on aarch64 without it), 1.18
  sets `cmake_policy(CMP0104 OLD)` which cmake 4 removed
  (`CMAKE_SPEC='cmake>=3.28,<4'`), FP8 has to be switched off explicitly
  (`DISABLE_TYPES=float8`) because the `CUDA_R_8F_E4M3`/`E5M2` enums only exist from
  CUDA 11.8 — FP8 tensor cores need sm_89 anyway, so Orin loses nothing — and the
  CUDA toolkit has to be upgraded to 11.8 (`CUDA_UPGRADE=11-8` with
  `CUDA_HOME_DIR=/usr/local/cuda-11.8`). 3.12 is the highest python 1.18 supports.
- **The CUDA 11.4 that L4T r35 ships cannot build onnxruntime at all.** Its nvcc
  miscompiles `resize_antialias_impl.cu`: cicc emits IR that fails its own verifier,
  ~90 × `Call parameter type does not match function signature`, on the `std::tuple`
  arguments of the `DISPATCH_ANTIALIAS_FILTER_SETUP` macro. It is an nvcc bug rather
  than a host-compiler one — microsoft/onnxruntime#20330 collects reports across
  CUDA 11.4/11.6 and gcc 9/10/12, including one from a JetPack 5.1.2 Xavier NX on
  the same gcc-10 we use. 11.8 is the earliest nvcc that compiles it. NVIDIA
  publishes an 11.8 toolkit for Tegra in the `ubuntu2004/arm64` repo (the "CUDA
  upgrade package"), which is what `CUDA_UPGRADE` installs; cuDNN deliberately stays
  at the base image's 8.x, so the wheel still links `libcudnn.so.8`.
- JP5 *can* run CUDA 12.2 via `cuda_compat`, but nixpkgs/jetpack pins JP5 to CUDA 11.4
  natively; relying on cuda_compat for a 12.x build is unproven here.

## Python versions

3.10, 3.11, 3.12 confirmed in nixpkgs unstable. 3.13 available. **3.14 is risky**
(very new; onnxruntime 1.26 may not build/declare support — treat as best-effort).

## Prioritized build list (Jeff's order)

Deliverable = Jetson (aarch64) wheels. Each needs a native aarch64 builder.

| # | Python | Target   | SM     | Status / notes |
|---|--------|----------|--------|----------------|
| 0 | 3.12   | x86_64   | sm_90* | **validation** on local RTX 5070 (PTX JIT). In progress. |
| 1 | 3.12   | JP6      | sm_87  | first real deliverable |
| 2 | 3.12   | JP5      | sm_87  | **shipped** — see below |
| 3 | 3.11   | JP6      | sm_87  | |
| 4 | 3.11   | JP5      | sm_87  | **shipped** |
| 5 | 3.10   | JP6      | sm_87  | |
| 6 | 3.10   | JP5      | sm_87  | **shipped** |
| 7 | 3.14   | JP6      | sm_87  | risky (3.14 + ort 1.26) |
| 8 | 3.14   | JP5      | —      | impossible: ort 1.18 caps at python 3.12 |
| + | 3.x    | JP7      | sm_110 | Thor — only if a Thor target is wanted |

\* x86_64 validation builds for sm_90 + PTX (nixpkgs CUDA < 12.8 has no sm_120 SASS;
compute_90 PTX JITs forward onto the 5070's sm_120).

### JP5 is shipped (2026-08-28)

`onnxruntime-gpu-extended==1.18.1.11.8`, cp38 through cp312, tagged
`manylinux2014_aarch64`. Built on L4T r35.1.0 with the CUDA 11.8 upgrade; links
`libcudnn.so.8` + `libcudart.so.11.0`. Verified on an Orin running JetPack 5.1.1:
`pip install onnxruntime-gpu-extended-auto` alone resolves to it and both a MatMul
(cuBLAS) and a Conv (cuDNN) run on the GPU.

One honest limit: **sm_72 / Xavier is not covered** — the build passes
`cuda_arch=87`, so these wheels are Orin-only despite JP5 also running on Xavier.
A follow-up batch built with `cuda_arch='72;87'` fixes that.

Flash and memory-efficient attention **are present**, contrary to what the CUDA 11.6
cutoff would suggest. `CUDA_UPGRADE=11-8` replaces the toolkit *before* cmake runs,
so onnxruntime sees 11.8 and enables them; `strings` on the shipped
`libonnxruntime_providers_cuda.so` finds ~11k `flash_fwd` instantiations. Note the
kernels are still guarded by `__CUDA_ARCH__ >= 800` internally, so they are live on
Orin (sm_87) and inert on Xavier (sm_72).

## Open decisions

- ~~**aarch64 builder**~~ — settled: GitHub `ubuntu-24.04-arm`. A JP5 matrix of five
  pythons builds in ~2h20m at `build_jobs=1` (serial is required; -j4 thrashes swap
  hard enough that GitHub kills the job).
- ~~JP5 onnxruntime version~~ — settled: 1.18.1.
- Whether to add `sm_72` so the JP5 wheels also cover Xavier.
- Whether Thor / JP7 is an actual target.
