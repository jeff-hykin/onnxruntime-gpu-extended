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
| 5.x     | r35   | 11.4 (→12.2 compat)| Orin (sm_87) + Xavier (**sm_72**) | needs older ort (~1.19/1.20, CUDA 11) ✗ |
| 7.x     | r38?  | 13.0              | Thor AGX (**sm_110 / cc 11.0**)   | 1.26.0 (if CUDA13 builds) |

Notes:
- **JP6 is the clean target**: CUDA 12.6 pairs with the onnxruntime 1.26.0 that
  nixpkgs already vendors (all FetchContent deps + a `dist` wheel output).
- **JP5 is extra work**: onnxruntime 1.26 requires CUDA 12; JP5 is CUDA 11.4. A JP5
  wheel needs an older onnxruntime branch (last CUDA-11 release, ~1.19/1.20) whose
  vendored deps differ — not reusable from nixpkgs' 1.26 derivation.
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
| 2 | 3.12   | JP5      | sm_87+sm_72 | needs older ort (CUDA 11) — blocked on ort version |
| 3 | 3.11   | JP6      | sm_87  | |
| 4 | 3.11   | JP5      | sm_87+sm_72 | older ort |
| 5 | 3.10   | JP6      | sm_87  | |
| 6 | 3.10   | JP5      | sm_87+sm_72 | older ort |
| 7 | 3.14   | JP6      | sm_87  | risky (3.14 + ort 1.26) |
| 8 | 3.14   | JP5      | sm_87+sm_72 | risky + older ort |
| + | 3.x    | JP7      | sm_110 | Thor — only if a Thor target is wanted |

\* x86_64 validation builds for sm_90 + PTX (nixpkgs CUDA < 12.8 has no sm_120 SASS;
compute_90 PTX JITs forward onto the 5070's sm_120).

## Open decisions

- **aarch64 builder**: reachable Orin over SSH, or GitHub `ubuntu-24.04-arm` runner?
  (Determines how Stage 2 actually runs.)
- JP5 onnxruntime version to track (CUDA-11 era) — only if JP5 wheels are required.
- Whether Thor / JP7 is an actual target.
