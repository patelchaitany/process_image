"""(Windows venv) ONNX files for the C++ engines: detector output kept channel-first [N,5,A] (the repo's final
Transpose to [N,A,5] is removed: K2 reads each channel with coalesced loads), recognisers with dynamic batch.
The EdgeFace files (models\\zoo, from github.com/yakhyo/edgeface-onnx) already have a dynamic batch."""
import os
import shutil
import onnx
from onnx import TensorProto, helper

M = r"E:\bench\models"
T = os.path.join(M, "tmp")
OUT = os.path.join(M, "cpp")
os.makedirs(OUT, exist_ok=True)


def strip_transpose(src, dst):
    m = onnx.load(src)
    g = m.graph
    out = g.output[0]
    node = [n for n in g.node if out.name in n.output][0]
    assert node.op_type == "Transpose", node.op_type
    A = out.type.tensor_type.shape.dim[1].dim_value
    g.node.remove(node)
    g.output.remove(out)
    g.output.append(helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ["N", 5, A]))
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print("det", dst, "A =", A)


for src, dst in (("yolo_r384.onnx", "det_r384.onnx"), ("yolo_r384_int8.onnx", "det_r384_int8.onnx")):
    if os.path.exists(os.path.join(T, src)):
        strip_transpose(os.path.join(T, src), os.path.join(OUT, dst))
for src, dst in (("rec_r50_dyn.onnx", "rec_r50.onnx"), ("rec_mbf_dyn.onnx", "rec_mbf.onnx"),
                 ("rec_r50_int8_raw.onnx", "rec_r50_int8.onnx"), ("rec_mbf_int8_raw.onnx", "rec_mbf_int8.onnx")):
    s = os.path.join(T, src)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(OUT, dst))
        m = onnx.load(s)
        print("rec", dst, "input", m.graph.input[0].name, "output", m.graph.output[0].name)
for src, dst in (("edgeface_s_gamma_05.onnx", "rec_edgeface_s.onnx"), ("edgeface_base.onnx", "rec_edgeface_base.onnx")):
    z = os.path.join(M, "zoo", src)
    if os.path.exists(z):
        shutil.copy(z, os.path.join(OUT, dst))
        m = onnx.load(z)
        print("rec", dst, "input", m.graph.input[0].name, "output", m.graph.output[0].name)
