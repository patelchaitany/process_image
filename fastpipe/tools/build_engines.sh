#!/bin/bash
# (WSL) Build TensorRT engines with FP16 I/O, a batch profile, and per-layer profiles for inspection.
set -e
M=/mnt/e/bench/models/cpp
E=${ENGINES:-/root/fastpipe/engines}
TRT=/usr/src/tensorrt/bin/trtexec
mkdir -p $E
build() {   # name onnx input item min opt max flags...
  local name=$1 onnx=$2 inp=$3 item=$4 lo=$5 opt=$6 hi=$7; shift 7
  [ -f "$onnx" ] || { echo "skip $name (no $onnx)"; return; }
  [ -f "$E/$name.plan" ] && { echo "have $name"; return; }
  echo "== building $name"
  $TRT --onnx=$onnx --saveEngine=$E/$name.plan --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw \
    --minShapes=$inp:${lo}x$item --optShapes=$inp:${opt}x$item --maxShapes=$inp:${hi}x$item \
    --shapes=$inp:${opt}x$item --builderOptimizationLevel=3 --profilingVerbosity=detailed \
    --dumpLayerInfo --exportLayerInfo=$E/$name.layers.json \
    --separateProfileRun --dumpProfile --exportProfile=$E/$name.profile.json "$@" > $E/$name.log 2>&1 \
    || { tail -20 $E/$name.log; return 1; }
  grep -E "Throughput|GPU Compute Time: min" $E/$name.log
}
REC_IN=$(python3 - <<'P' 2>/dev/null || echo input.1
import onnx; print(onnx.load("/mnt/e/bench/models/cpp/rec_r50.onnx").graph.input[0].name)
P
)
# Each recogniser's match thresholds, read by fastpipe from <engine>.thr. Cosine scores differ between models: these
# give every model the false-match rate ArcFace R50 has at 0.45 / 0.25 (benchmarks/models/RESULTS.md, "Match threshold").
thr() { [ -f "$E/$1.plan" ] && printf 'match_thr %s\nunknown_thr %s\n' "$2" "$3" > "$E/$1.thr"; return 0; }
build det_r384_fp16 $M/det_r384.onnx      images 3x384x640 1 16 32  --fp16
build rec_edgeface_s_fp16    $M/rec_edgeface_s.onnx    input 3x112x112 1 64 128 --fp16     # default recogniser
build rec_edgeface_base_fp16 $M/rec_edgeface_base.onnx input 3x112x112 1 64 128 --fp16
build rec_r50_fp16  $M/rec_r50.onnx       $REC_IN 3x112x112 1 64 128 --fp16
build rec_r50_int8  $M/rec_r50_int8.onnx  $REC_IN 3x112x112 1 64 128 --fp16 --int8
build rec_mbf_fp16  $M/rec_mbf.onnx       $REC_IN 3x112x112 1 64 128 --fp16
build rec_mbf_int8  $M/rec_mbf_int8.onnx  $REC_IN 3x112x112 1 64 128 --fp16 --int8
thr rec_edgeface_s_fp16    0.58 0.38
thr rec_edgeface_base_fp16 0.46 0.26
thr rec_r50_fp16           0.45 0.25
thr rec_r50_int8           0.45 0.25
thr rec_mbf_fp16           0.49 0.29
thr rec_mbf_int8           0.49 0.29
ls -la $E/*.plan
