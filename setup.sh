#!/bin/bash
set -euo pipefail

TRITON_CLIENT_VERSION="v2.68.0"
TRITON_CLIENT_TAR="${TRITON_CLIENT_VERSION}_ubuntu2404.clients.tar.gz"
TRITON_CLIENT_URL="https://github.com/triton-inference-server/server/releases/download/${TRITON_CLIENT_VERSION}/${TRITON_CLIENT_TAR}"
TRITON_INSTALL_DIR="/opt/tritonserver/client"

echo "==========================================="
echo "  process_image - Environment Setup"
echo "  Target: Ubuntu 24.04"
echo "==========================================="

# ── 1. Apt dependencies ──────────────────────────────────────────────
echo ""
echo "[1/3] Installing apt dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    python3 \
    python3-pip \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    libsqlite3-dev \
    libyaml-cpp-dev \
    libopencv-dev \
    libspdlog-dev \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    libcurl4-openssl-dev

# ── 2. Triton C++ Client SDK ─────────────────────────────────────────
echo ""
echo "[2/3] Installing Triton C++ Client SDK ${TRITON_CLIENT_VERSION}..."

if [ -d "${TRITON_INSTALL_DIR}" ] && [ -f "${TRITON_INSTALL_DIR}/include/grpc_client.h" ]; then
    echo "  Triton C++ Client SDK already installed at ${TRITON_INSTALL_DIR}, skipping."
else
    TMPDIR=$(mktemp -d)
    echo "  Downloading ${TRITON_CLIENT_TAR}..."
    wget -q --show-progress -O "${TMPDIR}/${TRITON_CLIENT_TAR}" "${TRITON_CLIENT_URL}"

    sudo mkdir -p "${TRITON_INSTALL_DIR}"
    sudo tar -xzf "${TMPDIR}/${TRITON_CLIENT_TAR}" -C "${TRITON_INSTALL_DIR}"
    rm -rf "${TMPDIR}"

    echo "${TRITON_INSTALL_DIR}/lib" | sudo tee /etc/ld.so.conf.d/triton-client.conf > /dev/null
    sudo ldconfig
    echo "  Installed to ${TRITON_INSTALL_DIR}"
fi

# ── 3. Python dependencies ───────────────────────────────────────────
echo ""
echo "[3/3] Installing Python dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip3 install --break-system-packages -r "${SCRIPT_DIR}/tools/requirements.txt"

# ── Done ──────────────────────────────────────────────────────────────
echo ""
echo "==========================================="
echo "  Setup complete!"
echo ""
echo "  Triton C++ SDK: ${TRITON_INSTALL_DIR}"
echo ""
echo "  To build:"
echo "    cmake -B build -DCMAKE_BUILD_TYPE=Release"
echo "    cmake --build build -j\$(nproc)"
echo "==========================================="
