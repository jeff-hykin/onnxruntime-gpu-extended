"""Run a matmul and a convolution on the Jetson GPU and check the results.

Builds tiny models in memory so nothing has to be downloaded, forces the CUDA
execution provider, and compares against a reference. Fails loudly if
onnxruntime silently falls back to CPU.

The Conv is the load-bearing half: MatMul dispatches to cuBLAS and passes even
when cuDNN is entirely unusable, which is how a wheel linked against the wrong
cuDNN major can look healthy. Only Conv actually exercises cuDNN.
"""

import numpy
from onnx import TensorProto, helper
import onnxruntime

print("python     ", __import__("sys").version.split()[0])
print("numpy      ", numpy.__version__)
print("onnxruntime", onnxruntime.__version__)
print("providers  ", onnxruntime.get_available_providers())

size = 256
graph = helper.make_graph(
    [helper.make_node("MatMul", ["left", "right"], ["product"])],
    "matmul",
    [
        helper.make_tensor_value_info("left", TensorProto.FLOAT, [size, size]),
        helper.make_tensor_value_info("right", TensorProto.FLOAT, [size, size]),
    ],
    [helper.make_tensor_value_info("product", TensorProto.FLOAT, [size, size])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9

session = onnxruntime.InferenceSession(
    model.SerializeToString(), providers=["CUDAExecutionProvider"]
)
used = session.get_providers()
print("session    ", used)
assert "CUDAExecutionProvider" in used, "CUDA provider not active — fell back to CPU"

generator = numpy.random.default_rng(0)
left = generator.random((size, size), dtype=numpy.float32)
right = generator.random((size, size), dtype=numpy.float32)
actual = session.run(None, {"left": left, "right": right})[0]

numpy.testing.assert_allclose(actual, left @ right, rtol=1e-3, atol=1e-3)
print("GPU MATMUL OK")

batch, in_channels, out_channels, side, kernel = 1, 3, 8, 32, 3
conv_graph = helper.make_graph(
    [helper.make_node("Conv", ["image", "kernel"], ["feature"], pads=[1, 1, 1, 1])],
    "conv",
    [
        helper.make_tensor_value_info(
            "image", TensorProto.FLOAT, [batch, in_channels, side, side]
        ),
        helper.make_tensor_value_info(
            "kernel", TensorProto.FLOAT, [out_channels, in_channels, kernel, kernel]
        ),
    ],
    [
        helper.make_tensor_value_info(
            "feature", TensorProto.FLOAT, [batch, out_channels, side, side]
        )
    ],
)
conv_model = helper.make_model(conv_graph, opset_imports=[helper.make_opsetid("", 13)])
conv_model.ir_version = 9

image = generator.random((batch, in_channels, side, side), dtype=numpy.float32)
kernel_weights = generator.random(
    (out_channels, in_channels, kernel, kernel), dtype=numpy.float32
)
feeds = {"image": image, "kernel": kernel_weights}

conv_session = onnxruntime.InferenceSession(
    conv_model.SerializeToString(), providers=["CUDAExecutionProvider"]
)
conv_used = conv_session.get_providers()
print("conv session", conv_used)
assert "CUDAExecutionProvider" in conv_used, "Conv fell back to CPU"

reference_session = onnxruntime.InferenceSession(
    conv_model.SerializeToString(), providers=["CPUExecutionProvider"]
)
numpy.testing.assert_allclose(
    conv_session.run(None, feeds)[0],
    reference_session.run(None, feeds)[0],
    rtol=1e-3,
    atol=1e-3,
)
print("GPU CONV OK (cuDNN)")
