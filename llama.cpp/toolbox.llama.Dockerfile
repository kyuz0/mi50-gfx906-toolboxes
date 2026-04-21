ARG ROCM_VERSION="7.2.1"
ARG BASE_TOOLBOX="docker.io/kyuz0/rocm-toolbox-gfx906:${ROCM_VERSION}"

# The source builder uses the exact same base toolbox
FROM ${BASE_TOOLBOX} AS builder
ARG ROCM_DOCKER_ARCH="gfx906"
ENV AMDGPU_TARGETS=${ROCM_DOCKER_ARCH}

RUN apt-get update && apt-get install -y \
    build-essential cmake git libssl-dev curl libgomp1

WORKDIR /app
# Context must be ./llama.cpp/
COPY ./submodules/llama.cpp .

# Compile Llama natively for gfx906 using the ROCm tools mapped into the host
RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DGGML_HIP=ON \
        -DGGML_HIP_ROCWMMA_FATTN=ON \
        -DAMDGPU_TARGETS="$ROCM_DOCKER_ARCH" \
        -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full

# Grab all the dynamic objects
RUN mkdir -p /app/lib-objects \
    && find build -name "*.so*" -exec cp -P {} /app/lib-objects/ \;


# Final Toolbox Stage
FROM ${BASE_TOOLBOX} AS final

# Install python and git for running requirements 
RUN apt-get update && apt-get install -y \
    git python3-pip python3 python3-wheel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Distribute llama.cpp artifacts into system folders for easy toolbox access
COPY --from=builder /app/full /usr/local/bin/
COPY --from=builder /app/lib-objects/ /usr/local/lib/

# Register the Shared Objects 
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf \
    && ldconfig

WORKDIR /usr/local/bin
RUN pip install --break-system-packages --upgrade setuptools \
    && pip install --break-system-packages -r requirements.txt \
    && pip install --break-system-packages questionary rich "huggingface_hub[cli]" hf_transfer

COPY ./hf_models.json /opt/hf_models.json
COPY ./get_models.py /usr/local/bin/get_models
RUN chmod +x /usr/local/bin/get_models

# Reset the entrypoint cleanly for toolbox shell behavior
ENTRYPOINT []
CMD ["/bin/bash"]
