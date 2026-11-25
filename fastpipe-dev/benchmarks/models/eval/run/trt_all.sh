#!/bin/bash
# (WSL) TensorRT FP16 speed of every candidate. Detectors: one 640x640 image. Recognisers: batch 64 of 112x112 where
# the file allows a batch, else batch 1. Output: /mnt/e/bench/data/trt_speed.log
M=/mnt/e/bench/models; Z=$M/zoo; T=/root/trt_speed.sh; L=/mnt/e/bench/data/trt_speed.log
: > $L
d() { bash $T "$1" "$2" 3x640x640 "$3" >> $L 2>&1; }
r() { bash $T "$1" "$2" 3x112x112 "$3" >> $L 2>&1; }
d $M/yolov8n-face.onnx images 0
d $Z/det_500m.onnx input.1 1
d $Z/scrfd_1g.onnx input.1 0
d $Z/det_2.5g.onnx input.1 1
d $Z/det_10g.onnx input.1 1
d $Z/scrfd_34g.onnx input.1 0
d $Z/face_detection_yunet_2023mar.onnx input 0
d $Z/yunet_n_640_640.onnx input 0
d $Z/retinaface_mv1_0.25.onnx input 1
d $Z/retinaface_mv2.onnx input 1
d $Z/yolov5n_face.onnx input 1
d $Z/yolov5s_face.onnx input 1
d $Z/aka_yolov8n-face.onnx images 0
d $Z/aka_yolov10n-face.onnx images 0
d $Z/aka_yolov11n-face.onnx images 0
d $Z/aka_yolov12n-face.onnx images 0
d $Z/centerface.onnx input 1
r $M/w600k_r50.onnx input.1 64
r $M/w600k_mbf.onnx input.1 64
r $Z/edgeface_xxs.onnx input 64
r $Z/edgeface_xs_gamma_06.onnx input 64
r $Z/edgeface_s_gamma_05.onnx input 64
r $Z/edgeface_base.onnx input 64
r $Z/adaface_ir_18.onnx input 64
r $Z/adaface_ir_50.onnx input 64
r $Z/auraface_r100.onnx input 64
r $Z/face_recognition_sface_2021dec.onnx data 0
r $Z/mobilenetv1_0.25.onnx input 0
r $Z/mobilenetv3_small.onnx input 0
r $Z/mobilenetv2.onnx input 0
r $Z/mobilenetv3_large.onnx input 0
echo ALLDONE >> $L
