#!/bin/bash
# (WSL, root) TensorRT 10.16 C++ SDK + trtexec from NVIDIA's Ubuntu 24.04 repo, pinned so only TensorRT comes from it.
set -e
dpkg --configure -a || true
if [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-trt.list ]; then
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" > /etc/apt/sources.list.d/cuda-ubuntu2404-trt.list
  printf 'Package: *\nPin: origin developer.download.nvidia.com\nPin: release l=NVIDIA CUDA\nPin-Priority: 100\n\nPackage: libnvinfer* libnvonnxparsers* libnvinfer-bin tensorrt* python3-libnvinfer*\nPin: origin developer.download.nvidia.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/trt-only
  apt-get update -qq
fi
V=10.16.1.11-1+cuda12.9
P="libnvinfer10 libnvinfer-dev libnvinfer-headers-dev libnvinfer-safe-headers-dev libnvinfer-plugin10 libnvinfer-plugin-dev libnvinfer-headers-plugin-dev libnvonnxparsers10 libnvonnxparsers-dev libnvinfer-bin libnvinfer-lean10 libnvinfer-dispatch10 libnvinfer-vc-plugin10"
A=""; for p in $P; do A="$A $p=$V"; done
for i in 1 2 3 4; do
  out=$(DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends $A 2>&1) && break
  miss=$(echo "$out" | grep -oE "Depends: [a-z0-9.+-]+ \(= $(echo $V | sed 's/+/\\+/')\)" | awk '{print $2}' | sort -u)
  [ -z "$miss" ] && { echo "$out" | tail -8; exit 1; }
  for x in $miss; do A="$A $x=$V"; done
done
echo "$out" | tail -3
apt-mark hold $P >/dev/null 2>&1 || true
ls -la /usr/src/tensorrt/bin/trtexec /usr/include/x86_64-linux-gnu/NvInfer.h
