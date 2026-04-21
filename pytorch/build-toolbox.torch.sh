#!/bin/bash
set -e
set -o pipefail

cd $(dirname $0)

# Import variables (handles setting ROCM_ARCH to gfx906, etc)
source ../env.sh "pytorch" "rocm"

# Apply preset environments if evaluating manual scripts
ROCM_VERSION=${TORCH_ROCM_VERSION:-"6.4.4"}
TORCH_VERSION=${TORCH_VERSION:-"v2.7.1"}

TOOLBOX_TAG="docker.io/kyuz0/pytorch-toolbox-gfx906:${TORCH_VERSION}-rocm-${ROCM_VERSION}"

echo "=================================================="
echo ">>> Building PyTorch Toolbox from Source..."
echo "    PyTorch: ${TORCH_VERSION}"
echo "    ROCm:    ${ROCM_VERSION}"
echo "    Target:  ${TOOLBOX_TAG}"
echo "=================================================="
echo "WARNING: Building PyTorch + Vision + Audio natively on Podman takes several hours."

mkdir -p ./logs

# We run podman natively, layering PyTorch compilation directly on top of the previously built rocm-toolbox container base!
# By targeting the `final` layer of torch.Dockerfile, the output retains all the dbus/sudo hacks of the base image.
podman build -t "${TOOLBOX_TAG}" \
  --build-arg BASE_ROCM_IMAGE="docker.io/kyuz0/rocm-toolbox-gfx906:${ROCM_VERSION}" \
  --build-arg ROCM_ARCH="gfx906" \
  --build-arg PYTORCH_BRANCH="${TORCH_VERSION}" \
  --build-arg PYTORCH_MAX_JOBS="${TORCH_MAX_JOBS:-$(nproc)}" \
  --build-arg PYTORCH_VISION_BRANCH="${TORCH_VISION_VERSION}" \
  --target final \
  -f ./torch.Dockerfile ./build-context \
  2>&1 | tee ./logs/build_torch_toolbox_$(date +%Y%m%d%H%M%S).log

echo ">>> Pushing ${TOOLBOX_TAG} to Docker Hub..."
podman push "${TOOLBOX_TAG}"

echo "Build and push complete! The PyTorch toolbox is ready and synced remotely: ${TOOLBOX_TAG}"
