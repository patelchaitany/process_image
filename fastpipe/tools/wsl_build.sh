#!/bin/bash
# (WSL) copy sources from E:\bench\fastpipe and build
set -e
mkdir -p /root/fastpipe
cp -r /mnt/e/bench/fastpipe/src /mnt/e/bench/fastpipe/tools /mnt/e/bench/fastpipe/CMakeLists.txt /root/fastpipe/
find /root/fastpipe/src /root/fastpipe/tools /root/fastpipe/CMakeLists.txt -type f -exec sed -i 's/\r$//' {} +
chmod +x /root/fastpipe/tools/*.sh
cd /root/fastpipe
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > build_cmake.log 2>&1 || { tail -20 build_cmake.log; exit 1; }
cmake --build build -j 12 2>&1 | grep -E "error|warning: |registers|smem|spill|Linking|Built" | grep -v "Wno" | head -60
ls -la build/fastpipe
