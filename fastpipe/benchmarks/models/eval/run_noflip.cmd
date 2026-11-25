@echo off
rem Recognisers without the mirror-image embedding (how the pipeline runs them), one process per model.
set PY=E:\bench\venv\Scripts\python.exe
cd /d E:\bench\eval
for %%K in ("arcface-r50-w600k (current)" "mobilefacenet-w600k" "edgeface-xxs" "edgeface-xs" "edgeface-s" "edgeface-base" "sface" "adaface-ir18" "adaface-ir50" "auraface-r100") do (
  %PY% eval_rec.py %%K --no-flip --sizes 28 --out E:\bench\data\rec_results_noflip.jsonl >> E:\bench\data\rec_noflip.log 2>&1
)
echo ALLDONE >> E:\bench\data\rec_noflip.log
