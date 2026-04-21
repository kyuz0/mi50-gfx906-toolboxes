#!/bin/bash
set -e
set -o pipefail

cd $(dirname $0)
source preset.rocm-7.2.1.sh || true # Provide fallback if empty
source ../env.sh "comfyui" "pytorch"

mkdir -p ./logs

TOOLBOX_TAG="docker.io/kyuz0/comfy-toolbox-gfx906:latest"
echo ">>> Building ComfyUI Toolbox: ${TOOLBOX_TAG}"

echo ">>> Staging TUI launcher scripts into build context..."
mkdir -p ./build-context
cp start_comfy.py 99-toolbox-banner.sh set_extra_paths.sh get_qwen_workflows.sh ./build-context/
cp -r workflows ./build-context/

podman build -t "${TOOLBOX_TAG}" \
  --build-arg BASE_PYTORCH_IMAGE=${TORCH_IMAGE}:${COMFYUI_PYTORCH_VERSION:-v2.7.1}-rocm-${COMFYUI_ROCM_VERSION:-7.2.1} \
  --progress=plain --target final -f ./toolbox.comfy.Dockerfile ./build-context 2>&1 | tee ./logs/build_$(date +%Y%m%d%H%M%S).log

echo ">>> Pushing ${TOOLBOX_TAG} to Docker Hub..."
podman push "${TOOLBOX_TAG}" || echo "Warning: Push failed, check credentials."

echo "Build and push complete! The ComfyUI toolbox is ready: ${TOOLBOX_TAG}"
