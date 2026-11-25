"""Make ORT's QDQ ResNet50 TensorRT-friendly: opset 13 (per-channel DQ axis), and fold INT32 bias DequantizeLinear
nodes (unsupported by TensorRT) into FP32 bias constants. Weights/activations stay INT8 QDQ."""
import numpy as np, onnx
from onnx import numpy_helper
m = onnx.load(r"E:\bench\models\tmp\rec_r50_int8_raw.onnx")
for o in m.opset_import:
    if o.domain in ("", "ai.onnx"): o.version = max(o.version, 13)
g = m.graph
inits = {t.name: t for t in g.initializer}
folded = 0
for n in list(g.node):
    if n.op_type != "DequantizeLinear" or n.input[0] not in inits: continue
    x = numpy_helper.to_array(inits[n.input[0]])
    if x.dtype != np.int32: continue
    sc = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
    zp = numpy_helper.to_array(inits[n.input[2]]).astype(np.float32) if len(n.input) > 2 and n.input[2] else 0.0
    f = ((x.astype(np.float32) - zp) * sc).astype(np.float32)
    g.initializer.append(numpy_helper.from_array(f, n.output[0]))
    g.node.remove(n)
    folded += 1
used = {i for n in g.node for i in n.input}
for t in list(g.initializer):
    if t.name not in used and t.name not in {o.name for o in g.output}: g.initializer.remove(t)
onnx.checker.check_model(m)
onnx.save(m, r"E:\bench\models\cpp\rec_r50_int8.onnx")
print("folded int32 bias DQ:", folded, "| Q", sum(n.op_type == "QuantizeLinear" for n in g.node), "DQ", sum(n.op_type == "DequantizeLinear" for n in g.node))
