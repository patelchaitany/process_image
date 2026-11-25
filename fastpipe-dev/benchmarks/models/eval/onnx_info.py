"""(Windows venv) print input name / shape of ONNX files, one JSON per line: used to build trtexec commands."""
import json, sys
import onnx
for p in sys.argv[1:]:
    try:
        m = onnx.load(p, load_external_data=False)
        init = {i.name for i in m.graph.initializer}
        i = [x for x in m.graph.input if x.name not in init][0]
        dims = [d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else -1 for d in i.type.tensor_type.shape.dim]
        print(json.dumps({"file": p.replace("\\", "/").split("/")[-1], "input": i.name, "dims": dims,
                          "opset": m.opset_import[0].version, "outputs": len(m.graph.output)}))
    except Exception as e:      # noqa: BLE001
        print(json.dumps({"file": p, "error": str(e)[:200]}))
