"""
onnxruntime-gpu-extended: Pre-built ONNX Runtime CUDA wheels for platforms
upstream onnxruntime-gpu does not ship (primarily NVIDIA Jetson / aarch64 + CUDA).

Real releases republish the upstream `onnxruntime-gpu` wheels plus Nix-built
Jetson wheels under this single name, keeping the internal `onnxruntime` package
intact so `import onnxruntime` works unchanged.

Usage:
    pip install onnxruntime-gpu-extended
    import onnxruntime
"""

__version__ = "0.0.1"
