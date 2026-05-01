# Build seq: rocm_base => build_base => build_triton => build_fa => build_vllm => final

ARG BASE_PYTORCH_IMAGE="docker.io/kyuz0/pytorch-toolbox-gfx906:v2.11.0-rocm-7.2.1"
ARG MAX_JOBS=""
ARG EXTRA_REQUIREMENTS="empty.txt"

ARG VLLM_REPO="https://github.com/ai-infos/vllm-gfx906-mobydick.git"
ARG VLLM_BRANCH="main"
ARG VLLM_COMMIT=""
ARG VLLM_PATCH="empty.patch"

ARG TRITON_REPO="https://github.com/ai-infos/triton-gfx906.git"
ARG TRITON_BRANCH="main"
ARG TRITON_COMMIT=""
ARG TRITON_PATCH="empty.patch"

ARG FA_REPO="https://github.com/ai-infos/flash-attention-gfx906.git"
ARG FA_BRANCH="main"
ARG FA_COMMIT=""
ARG FA_PATCH="empty.patch"

############# Base image #############
FROM ${BASE_PYTORCH_IMAGE} AS rocm_base

# Set environment variables
ENV PYTORCH_ROCM_ARCH=$ROCM_ARCH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:
ENV VLLM_TARGET_DEVICE=rocm
ENV FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=0
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
ENV PYTHONNOUSERSITE=1

# Install base tools
RUN pip3 install --upgrade --ignore-installed '/opt/rocm/share/amd_smi' pyjwt && \
    pip3 cache purge && \
    apt install curl git wget jq aria2 -y

############# Clone repos #############
FROM rocm_base AS files_triton
ARG TRITON_REPO
ARG TRITON_BRANCH
ARG TRITON_COMMIT
ARG TRITON_PATCH
# Clone
WORKDIR /app/triton
RUN git clone --recurse-submodules --shallow-submodules --jobs 4 --branch ${TRITON_BRANCH} ${TRITON_REPO} .
RUN if [ "$TRITON_COMMIT" != "" ]; then git checkout "$TRITON_COMMIT"; fi
# Patch
COPY ./patch/${TRITON_PATCH} ./${TRITON_PATCH}
RUN git apply ./${TRITON_PATCH} --allow-empty

FROM rocm_base AS files_fa
ARG FA_REPO
ARG FA_BRANCH
ARG FA_COMMIT
ARG FA_PATCH
# Clone
WORKDIR /app/flash-attention
RUN git clone --recurse-submodules --shallow-submodules --jobs 4 --branch ${FA_BRANCH} ${FA_REPO} .
RUN if [ "$FA_COMMIT" != "" ]; then git checkout "$FA_COMMIT"; fi
# Patch
COPY ./patch/${FA_PATCH} ./${FA_PATCH}
RUN git apply ./${FA_PATCH} --allow-empty

FROM rocm_base AS files_vllm
ARG VLLM_REPO
ARG VLLM_BRANCH
ARG VLLM_COMMIT
ARG VLLM_PATCH
# Clone
WORKDIR /app/vllm
RUN git clone --recurse-submodules --shallow-submodules --jobs 4 --branch ${VLLM_BRANCH} ${VLLM_REPO} .
RUN if [ "$VLLM_COMMIT" != "" ]; then git checkout "$VLLM_COMMIT"; fi
# Patch
COPY ./patch/${VLLM_PATCH} ./${VLLM_PATCH}
RUN git apply ./${VLLM_PATCH} --allow-empty
# RUN sed -i 's/gfx906/gfx900;gfx906/g' CMakeLists.txt || true

FROM rocm_base AS files_extra
ARG EXTRA_REQUIREMENTS
WORKDIR /app/extra-requirements
COPY ./requirements/${EXTRA_REQUIREMENTS} /app/extra-requirements/requirements.txt

############# Build base #############
FROM rocm_base AS build_base
RUN pip3 install build

############# Build triton #############
FROM build_base AS build_triton
COPY --from=files_triton /app/triton /app/triton
WORKDIR /app/triton
RUN pip3 install -r python/requirements.txt
# Build
ARG MAX_JOBS
RUN MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    python -m build --wheel --no-isolation --outdir /dist
RUN pip3 install /dist/triton-*.whl
RUN ls /dist

############# Build FA #############
FROM build_triton AS build_fa
COPY --from=files_fa /app/flash-attention /app/flash-attention
WORKDIR /app/flash-attention
RUN pip3 install ninja packaging wheel pybind11 psutil
# Build
ARG MAX_JOBS
RUN MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    python -m build --wheel --no-isolation --outdir /dist
RUN pip3 install /dist/flash_attn-*.whl
RUN ls /dist

############# Build vllm #############
FROM build_fa AS build_vllm
COPY --from=files_vllm /app/vllm /app/vllm
WORKDIR /app/vllm
RUN pip3 install -r requirements/rocm.txt
# Build
ARG MAX_JOBS
RUN MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    python -m build --wheel --no-isolation --outdir /dist
RUN pip3 install /dist/vllm-*.whl
RUN ls /dist 

############# Install all #############
FROM rocm_base AS final
WORKDIR /app/vllm
RUN --mount=type=bind,from=build_vllm,src=/app/vllm/requirements/,target=/app/vllm/requirements \
    --mount=type=bind,from=files_extra,src=/app/extra-requirements/,target=/app/extra-requirements \
    --mount=type=bind,from=build_vllm,src=/dist/,target=/dist \
    pip3 install /dist/*.whl -r /app/vllm/requirements/rocm.txt && \
    pip3 install -r /app/extra-requirements/*.txt && \
    pip3 cache purge && \
    true

# 🚨 Vega10 LDS fix: BLOCK_M 128→64 in the V1 attention Triton kernel.
# The default BLOCK_M=128 produces 81920 bytes of shared memory, exceeding
# gfx906's 64KB (65536 byte) LDS hardware limit. This halves it to ~49KB.
RUN sed -i 's/BLOCK_M = 128/BLOCK_M = 64  # PATCHED for Vega10 LDS limit/' \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/prefix_prefill.py

# Install utility packages
RUN apt-get update && apt-get install -y dialog vim pciutils && rm -rf /var/lib/apt/lists/*

# Install toolbox scripts
COPY start_vllm.py models.py run_vllm_bench_mi50.py 99-toolbox-banner.sh /opt/
RUN chmod +x /opt/start_vllm.py /opt/99-toolbox-banner.sh && \
    ln -s /opt/start_vllm.py /usr/local/bin/start-vllm && \
    ln -s /opt/run_vllm_bench_mi50.py /usr/local/bin/run-vllm-bench && \
    ln -s /opt/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh

CMD ["/bin/bash"]
