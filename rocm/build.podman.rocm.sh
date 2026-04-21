#!/bin/bash
set -e

cd $(dirname $0)
source ../env.sh "rocm"

IMAGE_TAGS=(
  "$PATCHED_ROCM_IMAGE:${ROCM_VERSION}-${REPO_GIT_REF}-complete"
  "$PATCHED_ROCM_IMAGE:${ROCM_VERSION}-complete"
)

# We removed the image push registry check since you are building purely locally.

DOCKER_EXTRA_ARGS=()
for (( i=0; i<${#IMAGE_TAGS[@]}; i++ )); do
  DOCKER_EXTRA_ARGS+=("-t" "${IMAGE_TAGS[$i]}")
done

mkdir ./logs || true

# Use Podman instead of Docker Buildx and do not push to registry.
echo "Building ROCm base image with Podman locally (not pushing)..."
podman build ${DOCKER_EXTRA_ARGS[@]} \
  --build-arg BASE_ROCM_IMAGE="${BASE_ROCM_IMAGE}:${ROCM_IMAGE_VER}-complete" \
  --build-arg ROCM_ARCH="${ROCM_ARCH}" \
  --target final -f ./rocm.Dockerfile ./submodules 2>&1 | tee ./logs/build_$(date +%Y%m%d%H%M%S).log

echo "Build complete! Image tagged locally."
