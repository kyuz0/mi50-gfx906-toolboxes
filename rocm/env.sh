#/bin/bash

pushd $(dirname ${BASH_SOURCE[0]})

# value from tag https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags e.g. 7.0/6.4.4
if [ "$ROCM_VERSION" == "" ]; then
  ROCM_VERSION=7.2.1
fi
if [ "$ROCM_IMAGE_VER" == "" ]; then
  ROCM_IMAGE_VER=7.2.1
fi

# target arch
if [ "$ROCM_ARCH" == "" ]; then
  ROCM_ARCH=gfx906
fi

# source image
if [ "$BASE_ROCM_IMAGE" == "" ]; then
  BASE_ROCM_IMAGE=docker.io/rocm/dev-ubuntu-24.04
fi

# destination image
if [ "$PATCHED_ROCM_IMAGE" == "" ]; then
  PATCHED_ROCM_IMAGE=docker.io/kyuz0/rocm-gfx906
fi

popd
