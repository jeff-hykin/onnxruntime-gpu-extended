"""Run a matmul on the Jetson GPU through onnxruntime and check the result.

Builds a tiny MatMul model in memory so nothing has to be downloaded, forces the
CUDA execution provider, and compares against numpy. Fails loudly if onnxruntime
silently falls back to CPU.
"""

import numpy
import onnx
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
