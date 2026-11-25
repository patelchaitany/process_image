@echo off
rem One process per model: a single process for several models ran out of memory.
cd /d E:\bench\eval
del E:\bench\data\thresholds.jsonl E:\bench\data\thresholds.log 2>nul
for %%K in ("arcface-r50-w600k (current)" "edgeface-s" "edgeface-base" "mobilefacenet-w600k") do (
  E:\bench\venv\Scripts\python.exe thresholds.py %%K >> E:\bench\data\thresholds.log 2>&1
)
echo ALLDONE >> E:\bench\data\thresholds.log
