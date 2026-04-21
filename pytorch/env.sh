#/bin/bash

pushd $(dirname ${BASH_SOURCE[0]})

# rocm version
if [ "$TORCH_ROCM_VERSION" == "" ];  then TORCH_ROCM_VERSION="7.2.1"; fi
# Limit max compilation jobs to avoid linux OOM killer (safely allocates ~64GB RAM footprint)
if [ "$TORCH_MAX_JOBS" == "" ];      then TORCH_MAX_JOBS="22"; fi
# torch git checkpoint
if [ "$TORCH_VERSION" == "" ];       then TORCH_VERSION="v2.11.0"; fi

# destination image
if [ "$TORCH_IMAGE" == "" ]; then
  TORCH_IMAGE="docker.io/kyuz0/pytorch-toolbox-gfx906"
fi

popd
