@echo off
rem One process per model, one at a time: a single process for all models ran out of memory.
set PY=E:\bench\venv\Scripts\python.exe
cd /d E:\bench\eval
for %%K in ("scrfd-10g" "scrfd-34g" "yunet-s" "yunet-n" "retinaface-mnet0.25" "retinaface-mnetv2" "yolov5n-face" "yolov5s-face" "yolov8n-face (akanametov)" "yolov10n-face" "yolov11n-face" "yolov12n-face" "centerface") do (
  %PY% eval_det.py %%K >> E:\bench\data\det_run2.log 2>&1
)
for %%K in ("adaface-ir18" "adaface-ir50" "auraface-r100" "sface") do (
  %PY% eval_rec.py %%K >> E:\bench\data\rec_run2.log 2>&1
)
echo ALLDONE >> E:\bench\data\det_run2.log
