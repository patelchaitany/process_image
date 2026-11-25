@echo off
rem The detectors that accept any input size, at long side 1280 (the main run uses 640).
set PY=E:\bench\venv\Scripts\python.exe
cd /d E:\bench\eval
for %%K in ("scrfd-2.5g" "scrfd-10g" "scrfd-500m") do (
  %PY% eval_det.py %%K --size 1280 --out E:\bench\data\det_results_size.jsonl >> E:\bench\data\det_size.log 2>&1
)
echo ALLDONE >> E:\bench\data\det_size.log
