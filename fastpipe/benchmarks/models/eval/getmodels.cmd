@echo off
rem Download the candidate models (ONNX only) into E:\bench\models\zoo
cd /d E:\bench\models\zoo
set U=https://github.com/yakhyo/uniface/releases/download/weights
curl.exe -sL -o det_2.5g.onnx https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx
curl.exe -sL -o det_10g.onnx %U%/scrfd_10g_kps.onnx
curl.exe -sL -o scrfd_34g.onnx https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX/resolve/main/34g.onnx
curl.exe -sL -o scrfd_1g.onnx https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX/resolve/main/1g.onnx
curl.exe -sL -o face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl.exe -sL -o yunet_n_640_640.onnx https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/onnx/yunet_n_640_640.onnx
curl.exe -sL -o retinaface_mv1_0.25.onnx %U%/retinaface_mv1_0.25.onnx
curl.exe -sL -o retinaface_mv2.onnx %U%/retinaface_mv2.onnx
curl.exe -sL -o yolov5n_face.onnx https://github.com/yakhyo/yolov5-face-onnx-inference/releases/download/weights/yolov5n_face.onnx
curl.exe -sL -o yolov5s_face.onnx https://github.com/yakhyo/yolov5-face-onnx-inference/releases/download/weights/yolov5s_face.onnx
for %%m in (yolov8n yolov10n yolov11n yolov12n) do curl.exe -sL -o aka_%%m-face.onnx https://github.com/akanametov/yolo-face/releases/download/1.0.0/%%m-face.onnx
curl.exe -sL -o centerface.onnx %U%/centerface.onnx
for %%m in (edgeface_xxs edgeface_xs_gamma_06 edgeface_s_gamma_05 edgeface_base) do curl.exe -sL -o %%m.onnx https://github.com/yakhyo/edgeface-onnx/releases/download/weights/%%m.onnx
for %%m in (adaface_ir_18 adaface_ir_50) do curl.exe -sL -o %%m.onnx https://github.com/yakhyo/adaface-onnx/releases/download/weights/%%m.onnx
curl.exe -sL -o face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
for %%m in (mobilenetv1_0.25 mobilenetv2 mobilenetv3_small mobilenetv3_large) do curl.exe -sL -o %%m.onnx %U%/%%m.onnx
curl.exe -sL -o auraface_r100.onnx https://github.com/yakhyo/auraface-onnx/releases/download/weights/auraface_r100.onnx
cd /d E:\bench\data\pairs
if not exist agedb_30.bin curl.exe -sL -o agedb_30.bin https://huggingface.co/datasets/gaunernst/face-recognition-eval/resolve/main/agedb_30.bin
dir /b E:\bench\data\pairs\*.bin
E:\bench\venv\Scripts\python.exe E:\bench\eval\onnx_info.py E:\bench\models\zoo\det_500m.onnx E:\bench\models\zoo\det_2.5g.onnx E:\bench\models\zoo\det_10g.onnx E:\bench\models\zoo\scrfd_34g.onnx E:\bench\models\zoo\scrfd_1g.onnx E:\bench\models\zoo\face_detection_yunet_2023mar.onnx E:\bench\models\zoo\yunet_n_640_640.onnx E:\bench\models\zoo\retinaface_mv1_0.25.onnx E:\bench\models\zoo\retinaface_mv2.onnx E:\bench\models\zoo\yolov5n_face.onnx E:\bench\models\zoo\yolov5s_face.onnx E:\bench\models\zoo\aka_yolov8n-face.onnx E:\bench\models\zoo\aka_yolov10n-face.onnx E:\bench\models\zoo\aka_yolov11n-face.onnx E:\bench\models\zoo\aka_yolov12n-face.onnx E:\bench\models\zoo\centerface.onnx E:\bench\models\zoo\edgeface_xxs.onnx E:\bench\models\zoo\edgeface_xs_gamma_06.onnx E:\bench\models\zoo\edgeface_s_gamma_05.onnx E:\bench\models\zoo\edgeface_base.onnx E:\bench\models\zoo\adaface_ir_18.onnx E:\bench\models\zoo\adaface_ir_50.onnx E:\bench\models\zoo\face_recognition_sface_2021dec.onnx E:\bench\models\zoo\mobilenetv1_0.25.onnx E:\bench\models\zoo\mobilenetv2.onnx E:\bench\models\zoo\mobilenetv3_small.onnx E:\bench\models\zoo\mobilenetv3_large.onnx E:\bench\models\zoo\auraface_r100.onnx E:\bench\models\yolov8n-face.onnx E:\bench\models\w600k_r50.onnx E:\bench\models\w600k_mbf.onnx
