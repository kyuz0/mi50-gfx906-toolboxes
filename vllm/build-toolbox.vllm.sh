#/bin/bash
set -e
set -o pipefail

cd $(dirname $0)
source preset.0.19.1-rocm-7.2.1-aiinfos.sh
source ../env.sh "vllm-v2" "pytorch"

# legacy push check stripped out

mkdir -p ./logs

TOOLBOX_TAG="docker.io/kyuz0/vllm-toolbox-gfx906:${VLLM_PRESET_NAME}"
echo ">>> Building vLLM Toolbox: ${TOOLBOX_TAG}"
echo ">>> Staging TUI launcher scripts into build context..."
mkdir -p ./build-context
cp start_vllm.py models.py run_vllm_bench_mi50.py 99-toolbox-banner.sh ./build-context/

podman build -t "${TOOLBOX_TAG}" \
  --build-arg BASE_PYTORCH_IMAGE=${TORCH_IMAGE}:${VLLM_PYTORCH_VERSION}-rocm-${VLLM_ROCM_VERSION} \
  --build-arg MAX_JOBS="${VLLM_MAX_JOBS}" \
  --build-arg EXTRA_REQUIREMENTS="${VLLM_EXTRA_REQUIREMENTS}" \
  \
  --build-arg VLLM_REPO=${VLLM_REPO}     \
  --build-arg VLLM_BRANCH=${VLLM_BRANCH} \
  --build-arg VLLM_COMMIT=${VLLM_COMMIT} \
  --build-arg VLLM_PATCH=${VLLM_PATCH}   \
  \
  --build-arg FA_REPO=${VLLM_FA_REPO}     \
  --build-arg FA_BRANCH=${VLLM_FA_BRANCH} \
  --build-arg FA_COMMIT=${VLLM_FA_COMMIT} \
  --build-arg FA_PATCH=${VLLM_FA_PATCH}   \
  \
  --build-arg TRITON_REPO=${VLLM_TRITON_REPO}     \
  --build-arg TRITON_BRANCH=${VLLM_TRITON_BRANCH} \
  --build-arg TRITON_COMMIT=${VLLM_TRITON_COMMIT} \
  --build-arg TRITON_PATCH=${VLLM_TRITON_PATCH}   \
  \
  --target final -f ./toolbox.vllm.Dockerfile ./build-context 2>&1 | tee ./logs/build_$(date +%Y%m%d%H%M%S).log

echo ">>> Pushing ${TOOLBOX_TAG} to Docker Hub..."
podman push "${TOOLBOX_TAG}"

echo "Build and push complete! The vLLM toolbox is ready: ${TOOLBOX_TAG}"
