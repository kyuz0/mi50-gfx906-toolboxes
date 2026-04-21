#!/bin/bash
set -e

cd $(dirname $0)

# Default to 6.4.4 as requested, or let the user override it via environment variables (e.g. ROCM_VERSION=7.2.1 ./build-toolbox.llama.sh)
ROCM_VERSION=${ROCM_VERSION:-"6.4.4"}
TOOLBOX_TAG="docker.io/kyuz0/llamacpp-toolbox-gfx906:rocm-${ROCM_VERSION}"

mkdir -p logs

echo "=================================================="
echo ">>> Building Llama.cpp Toolbox for ROCm $ROCM_VERSION locally..."
echo "=================================================="

# We execute podman build within the llama.cpp/ context
podman build -t ${TOOLBOX_TAG} \
  --build-arg ROCM_VERSION=${ROCM_VERSION} \
  --build-arg ROCM_DOCKER_ARCH=gfx906 \
  -f ./toolbox.llama.Dockerfile . 2>&1 | tee ./logs/build_llama_toolbox_${ROCM_VERSION}_$(date +%Y%m%d%H%M%S).log

echo "Build complete! The image has been tagged locally as: ${TOOLBOX_TAG}"
