#!/bin/bash
# (WSL) usage: exp_build_wsl.sh <name>   builds E:\bench\exp\<name>\ into /root/fp-<name>/build
set -e
name=$1; d=/root/fp-$name
mkdir -p $d && rm -rf $d/src $d/CMakeLists.txt
cp -r /mnt/e/bench/exp/$name/src /mnt/e/bench/exp/$name/CMakeLists.txt $d/
find $d/src $d/CMakeLists.txt -type f -exec sed -i 's/\r$//' {} +
cd $d && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > build_cmake.log 2>&1 || { tail -20 build_cmake.log; exit 1; }
cmake --build build -j 8 2>&1 | grep -E "error|Error|Built target|undefined" | head -40
ls -la build/fastpipe
