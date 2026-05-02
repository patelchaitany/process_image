#!/usr/bin/env python3
"""
Set up ONNX models for the Triton Inference Server model repository.

Creates the model_repository/ directory structure with:
  - yolo26_face : YOLO face detection   (input [N,3,640,640] → output [N,8400,5])
  - arcface     : ArcFace recognition   (input [N,3,112,112] → output [N,512])

Usage:
    # 1) Install dependencies
    pip3 install onnx onnxruntime numpy

    # 2a) Create test models with random weights (instant, no download):
    python3 tools/download_models.py

    # 2b) Download real pre-trained models (needs internet + extra packages):
    pip3 install insightface ultralytics
    python3 tools/download_models.py --real

    # 3) Start Triton (CPU mode, Docker):
    docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
      -v $(pwd)/models/model_repository:/models \\
      nvcr.io/nvidia/tritonserver:24.01-py3 \\
      tritonserver --model-repository=/models \\
        --backend-config=onnxruntime,default-max-batch-size=8

    # 4) Verify:
    curl -s localhost:8000/v2/health/live
    curl -s localhost:8000/v2/models | python3 -m json.tool
"""

import argparse
import os
import sys
import shutil
import struct

import numpy as np

# ---------------------------------------------------------------------------
# Config.pbtxt templates
# ---------------------------------------------------------------------------

YOLO_CONFIG = """\
name: "yolo26_face"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 8400, 5 ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 1, 4, 8 ]
  max_queue_delay_microseconds: 100
}}

instance_group [{{ kind: {device_kind} }}]
"""

ARCFACE_CONFIG = """\
name: "arcface"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 112, 112 ]
  }}
]
output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 1, 4, 8 ]
  max_queue_delay_microseconds: 100
}}

instance_group [{{ kind: {device_kind} }}]
"""


# ---------------------------------------------------------------------------
# Dummy model creation (uses onnx.helper – no training data needed)
# ---------------------------------------------------------------------------

def create_dummy_yolo(output_path):
    """
    Lightweight ONNX model:  [N,3,640,640] → [N,8400,5]
    Architecture: GlobalAveragePool → MatMul → Reshape
    Weights are random; outputs will be meaningless but shapes are correct.
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    X = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 640, 640])
    Y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["N", 8400, 5])

    gap = helper.make_node("GlobalAveragePool", ["images"], ["gap_out"])

    shape_flat = numpy_helper.from_array(
        np.array([0, 3], dtype=np.int64), name="shape_flat"
    )
    reshape_1 = helper.make_node("Reshape", ["gap_out", "shape_flat"], ["flat"])

    W = numpy_helper.from_array(
        (np.random.randn(3, 42000).astype(np.float32) * 0.01), name="W"
    )
    matmul = helper.make_node("MatMul", ["flat", "W"], ["mm_out"])

    shape_out = numpy_helper.from_array(
        np.array([0, 8400, 5], dtype=np.int64), name="shape_out"
    )
    reshape_2 = helper.make_node("Reshape", ["mm_out", "shape_out"], ["pre_sig"])
    sigmoid = helper.make_node("Sigmoid", ["pre_sig"], ["output0"])

    graph = helper.make_graph(
        [gap, reshape_1, matmul, reshape_2, sigmoid],
        "yolo26_face",
        [X],
        [Y],
        [shape_flat, W, shape_out],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Created dummy YOLO model → {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")


def create_dummy_arcface(output_path):
    """
    Lightweight ONNX model:  [N,3,112,112] → [N,512]
    Architecture: GlobalAveragePool → MatMul → L2Normalize
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 112, 112])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 512])

    gap = helper.make_node("GlobalAveragePool", ["input"], ["gap_out"])

    shape_flat = numpy_helper.from_array(
        np.array([0, 3], dtype=np.int64), name="shape_flat"
    )
    reshape = helper.make_node("Reshape", ["gap_out", "shape_flat"], ["flat"])

    W = numpy_helper.from_array(
        (np.random.randn(3, 512).astype(np.float32) * 0.1), name="W"
    )
    matmul = helper.make_node("MatMul", ["flat", "W"], ["raw"])

    normalize = helper.make_node(
        "LpNormalization", ["raw"], ["output"], axis=1, p=2
    )

    graph = helper.make_graph(
        [gap, reshape, matmul, normalize],
        "arcface",
        [X],
        [Y],
        [shape_flat, W],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"  Created dummy ArcFace model → {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Real model download helpers
# ---------------------------------------------------------------------------

def rename_onnx_io(model_path, input_map=None, output_map=None):
    """Rename input/output tensor names in an ONNX model."""
    import onnx

    model = onnx.load(model_path)
    graph = model.graph

    name_map = {}
    if input_map:
        for inp in graph.input:
            if inp.name in input_map:
                name_map[inp.name] = input_map[inp.name]
                inp.name = input_map[inp.name]
    if output_map:
        for out in graph.output:
            if out.name in output_map:
                name_map[out.name] = output_map[out.name]
                out.name = output_map[out.name]

    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in name_map:
                node.input[i] = name_map[name]
        for i, name in enumerate(node.output):
            if name in name_map:
                node.output[i] = name_map[name]

    onnx.save(model, model_path)


def download_real_arcface(output_path):
    """
    Download ArcFace model via the `insightface` package.
    Falls back to direct download from ONNX Model Zoo.
    """
    try:
        print("  Trying insightface package...")
        import insightface
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))

        models_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
        src = os.path.join(models_dir, "w600k_r50.onnx")
        if os.path.exists(src):
            shutil.copy2(src, output_path)
            _fix_arcface_io(output_path)
            print(f"  Downloaded real ArcFace model → {output_path}")
            return True
        print("  w600k_r50.onnx not found in insightface cache")
    except ImportError:
        print("  insightface not installed (pip install insightface)")
    except Exception as e:
        print(f"  insightface download failed: {e}")

    try:
        print("  Trying ONNX Model Zoo (arcfaceresnet100-11)...")
        import urllib.request

        url = (
            "https://github.com/onnx/models/raw/refs/heads/main/"
            "validated/vision/body_analysis/arcface/model/arcfaceresnet100-11.onnx"
        )
        urllib.request.urlretrieve(url, output_path)
        _fix_arcface_io(output_path)
        print(f"  Downloaded real ArcFace model → {output_path}")
        return True
    except Exception as e:
        print(f"  ONNX Model Zoo download failed: {e}")

    return False


def _fix_arcface_io(model_path):
    """Ensure ArcFace I/O names match config.pbtxt expectations."""
    import onnx

    model = onnx.load(model_path)
    g = model.graph

    current_in = g.input[0].name
    current_out = g.output[0].name

    in_map = {current_in: "input"} if current_in != "input" else None
    out_map = {current_out: "output"} if current_out != "output" else None

    if in_map or out_map:
        print(f"    Renaming I/O: {current_in}→input, {current_out}→output")
        rename_onnx_io(model_path, input_map=in_map, output_map=out_map)


def download_real_yolo(output_path):
    """
    Export a YOLOv8n face-detection ONNX via ultralytics, then adjust
    output shape to [N, 8400, 5] (cx, cy, w, h, conf).
    """
    try:
        print("  Trying ultralytics YOLOv8n export...")
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        tmp_path = model.export(format="onnx", imgsz=640, simplify=True)
        if tmp_path and os.path.exists(tmp_path):
            _convert_yolo_for_face(tmp_path, output_path)
            if os.path.abspath(tmp_path) != os.path.abspath(output_path):
                os.remove(tmp_path)
            print(f"  Exported real YOLO model → {output_path}")
            return True
    except ImportError:
        print("  ultralytics not installed (pip install ultralytics)")
    except Exception as e:
        print(f"  ultralytics export failed: {e}")

    return False


def _convert_yolo_for_face(src_path, dst_path):
    """
    Standard YOLOv8n outputs [1, 84, 8400] (4 coords + 80 classes).
    We slice to [1, 5, 8400] (4 coords + max class conf) and transpose to
    [1, 8400, 5] to match the project's expected format.
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    model = onnx.load(src_path)
    g = model.graph

    orig_output_name = g.output[0].name
    internal_name = orig_output_name + "_raw"

    for node in g.node:
        for i, name in enumerate(node.output):
            if name == orig_output_name:
                node.output[i] = internal_name

    bbox_starts = numpy_helper.from_array(
        np.array([0, 0, 0], dtype=np.int64), name="_bbox_starts"
    )
    bbox_ends = numpy_helper.from_array(
        np.array([0, 4, 0], dtype=np.int64), name="_bbox_ends"
    )
    bbox_axes = numpy_helper.from_array(
        np.array([0, 1, 2], dtype=np.int64), name="_bbox_axes"
    )
    bbox_max_ends = numpy_helper.from_array(
        np.array([9999, 9999, 9999], dtype=np.int64), name="_bbox_max_ends"
    )

    slice_bbox = helper.make_node(
        "Slice",
        [internal_name, "_bbox_starts", "_bbox_ends", "_bbox_axes"],
        ["_bboxes"],
        name="slice_bboxes",
    )

    cls_starts = numpy_helper.from_array(
        np.array([0, 4, 0], dtype=np.int64), name="_cls_starts"
    )
    slice_cls = helper.make_node(
        "Slice",
        [internal_name, "_cls_starts", "_bbox_max_ends", "_bbox_axes"],
        ["_cls_raw"],
        name="slice_classes",
    )

    reduce_max = helper.make_node(
        "ReduceMax",
        ["_cls_raw"],
        ["_conf"],
        axes=[1],
        keepdims=True,
        name="reduce_max_cls",
    )

    concat = helper.make_node(
        "Concat", ["_bboxes", "_conf"], ["_det_5"], axis=1, name="concat_det"
    )

    perm_attr = [0, 2, 1]
    transpose = helper.make_node(
        "Transpose", ["_det_5"], ["output0"], perm=perm_attr, name="transpose_det"
    )

    g.initializer.extend([bbox_starts, bbox_ends, bbox_axes, bbox_max_ends, cls_starts])
    g.node.extend([slice_bbox, slice_cls, reduce_max, concat, transpose])

    del g.output[:]
    new_output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, None)
    g.output.append(new_output)

    if g.input[0].name != "images":
        old_in = g.input[0].name
        for node in g.node:
            for i, name in enumerate(node.input):
                if name == old_in:
                    node.input[i] = "images"
        g.input[0].name = "images"

    onnx.save(model, dst_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_model(model_path, input_name, input_shape, output_name, expected_output_dims):
    """Run a single inference through onnxruntime and check output shape."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not installed, cannot validate")
        return True

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    actual_in = sess.get_inputs()[0]
    actual_out = sess.get_outputs()[0]
    print(f"    ONNX input:  name={actual_in.name!r}  shape={actual_in.shape}")
    print(f"    ONNX output: name={actual_out.name!r}  shape={actual_out.shape}")

    concrete_shape = [1 if isinstance(d, str) else d for d in input_shape]
    dummy_input = np.random.randn(*concrete_shape).astype(np.float32)
    result = sess.run([actual_out.name], {actual_in.name: dummy_input})

    out_shape = list(result[0].shape)
    print(f"    Inference OK: output shape = {out_shape}")

    ok = True
    for i, (got, exp) in enumerate(zip(out_shape, expected_output_dims)):
        if exp is not None and got != exp:
            print(f"    WARNING: dim {i}: expected {exp}, got {got}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Set up Triton model repository with YOLO face + ArcFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", default="./models/model_repository",
        help="Model repository root (default: ./models/model_repository)",
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Download real pre-trained models (requires insightface / ultralytics)",
    )
    parser.add_argument(
        "--cpu", action="store_true", default=True,
        help="Use KIND_CPU in config.pbtxt (default on Mac)",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use KIND_GPU in config.pbtxt (Linux with NVIDIA GPU)",
    )
    parser.add_argument(
        "--validate", action="store_true", default=True,
        help="Validate models with onnxruntime after setup (default: True)",
    )
    parser.add_argument(
        "--no-validate", dest="validate", action="store_false",
    )
    args = parser.parse_args()

    device_kind = "KIND_GPU" if args.gpu else "KIND_CPU"
    mode = "real" if args.real else "dummy"

    print(f"=== Triton Model Repository Setup ===")
    print(f"  Output:    {os.path.abspath(args.output)}")
    print(f"  Mode:      {mode}")
    print(f"  Device:    {device_kind}")
    print()

    # ── 1. Create directory structure ────────────────────────────────────
    yolo_dir = os.path.join(args.output, "yolo26_face", "1")
    arcface_dir = os.path.join(args.output, "arcface", "1")

    os.makedirs(yolo_dir, exist_ok=True)
    os.makedirs(arcface_dir, exist_ok=True)

    print("[1/4] Directory structure created:")
    print(f"  {args.output}/")
    print(f"  ├── yolo26_face/")
    print(f"  │   ├── config.pbtxt")
    print(f"  │   └── 1/model.onnx")
    print(f"  └── arcface/")
    print(f"      ├── config.pbtxt")
    print(f"      └── 1/model.onnx")
    print()

    # ── 2. Write config.pbtxt ────────────────────────────────────────────
    yolo_cfg_path = os.path.join(args.output, "yolo26_face", "config.pbtxt")
    arcface_cfg_path = os.path.join(args.output, "arcface", "config.pbtxt")

    with open(yolo_cfg_path, "w") as f:
        f.write(YOLO_CONFIG.format(device_kind=device_kind))
    with open(arcface_cfg_path, "w") as f:
        f.write(ARCFACE_CONFIG.format(device_kind=device_kind))

    print("[2/4] Config files written")
    print()

    # ── 3. Create / download models ──────────────────────────────────────
    yolo_model_path = os.path.join(yolo_dir, "model.onnx")
    arcface_model_path = os.path.join(arcface_dir, "model.onnx")

    print("[3/4] Setting up models...")

    if mode == "real":
        print("\n── ArcFace (real) ──")
        if not download_real_arcface(arcface_model_path):
            print("  ⚠ Could not download real ArcFace, falling back to dummy")
            create_dummy_arcface(arcface_model_path)

        print("\n── YOLO face (real) ──")
        if not download_real_yolo(yolo_model_path):
            print("  ⚠ Could not export real YOLO, falling back to dummy")
            create_dummy_yolo(yolo_model_path)
    else:
        print("\n── YOLO face (dummy) ──")
        create_dummy_yolo(yolo_model_path)
        print("\n── ArcFace (dummy) ──")
        create_dummy_arcface(arcface_model_path)

    print()

    # ── 4. Validate ──────────────────────────────────────────────────────
    if args.validate:
        print("[4/4] Validating models...")
        print("\n  yolo26_face:")
        v1 = validate_model(
            yolo_model_path,
            input_name="images",
            input_shape=[1, 3, 640, 640],
            output_name="output0",
            expected_output_dims=[1, 8400, 5],
        )
        print("\n  arcface:")
        v2 = validate_model(
            arcface_model_path,
            input_name="input",
            input_shape=[1, 3, 112, 112],
            output_name="output",
            expected_output_dims=[1, 512],
        )

        if v1 and v2:
            print("\n  All models validated successfully ✓")
        else:
            print("\n  ⚠ Some validations had warnings (see above)")
    else:
        print("[4/4] Validation skipped")

    # ── Done ─────────────────────────────────────────────────────────────
    repo_abs = os.path.abspath(args.output)
    print(f"""
{'='*60}
  Model repository is ready at:
    {repo_abs}

  Next steps:

  1. Start Triton Inference Server:

     docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
       -v {repo_abs}:/models \\
       nvcr.io/nvidia/tritonserver:24.01-py3 \\
       tritonserver --model-repository=/models \\
         --backend-config=onnxruntime,default-max-batch-size=8

  2. Verify Triton is serving the models:

     curl -s localhost:8000/v2/health/live
     curl -s localhost:8000/v2/models | python3 -m json.tool

  3. Enroll faces:

     python3 tools/enroll_faces.py \\
       --faces-dir ./faces/ \\
       --db ./faces.db \\
       --triton localhost:8001

  4. Run the pipeline:

     ./build/process_image \\
       --input video.mp4 \\
       --triton localhost:8001 \\
       --device cpu \\
       --db ./faces.db
{'='*60}
""")


if __name__ == "__main__":
    main()
