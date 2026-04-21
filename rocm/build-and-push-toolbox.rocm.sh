#!/bin/bash
set -e

cd $(dirname $0)

VERSIONS=("6.4.4" "7.2.1")
USERNAME="kyuz0"

mkdir -p ./logs

for VERSION in "${VERSIONS[@]}"; do
  echo "=================================================="
  echo ">>> Generating Toolbox for ROCm version: $VERSION "
  echo "=================================================="
  
  # Load the preset
  source preset.rocm-${VERSION}.sh
  source ../env.sh "rocm"

  # Build the base image locally first (so it's guaranteed to be up-to-date and patched for gfx906)
  # This delegates to the script we made earlier
  ./build.podman.rocm.sh

  # Now wrap the base image with the Toolbox extensions
  BASE_IMAGE="${PATCHED_ROCM_IMAGE}:${ROCM_VERSION}-complete"
  TOOLBOX_TAG="docker.io/${USERNAME}/rocm-toolbox-gfx906:${ROCM_VERSION}"

  echo ">>> Building Toolbox container..."
  podman build -t ${TOOLBOX_TAG} \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    -f ./toolbox.rocm.Dockerfile . 2>&1 | tee ./logs/build_toolbox_${VERSION}_$(date +%Y%m%d%H%M%S).log

  echo ">>> Pushing ${TOOLBOX_TAG} to Docker Hub..."
  podman push ${TOOLBOX_TAG}
done

echo "All complete! Successfully compiled and pushed ROCm toolboxes to Docker Hub."
