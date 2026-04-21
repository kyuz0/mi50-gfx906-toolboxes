ARG BASE_PYTORCH_IMAGE
FROM ${BASE_PYTORCH_IMAGE} AS final

WORKDIR /opt

# Core packages for UI and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git ffmpeg vim dialog pciutils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ComfyUI Checkout
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI
WORKDIR /opt/ComfyUI

# Python dependencies
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir pillow opencv-python-headless imageio imageio-ffmpeg scipy "huggingface_hub[hf_transfer]" pyyaml websocket-client

# Essential Custom Nodes
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git && \
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    pip3 install --no-cache-dir gguf numpy

WORKDIR /opt

# Install utilities and banner
COPY start_comfy.py 99-toolbox-banner.sh set_extra_paths.sh get_qwen_workflows.sh /opt/
RUN mkdir -p /opt/ComfyUI/user/default/workflows/
COPY workflows/*.json /opt/ComfyUI/user/default/workflows/
RUN chmod +x /opt/start_comfy.py /opt/99-toolbox-banner.sh /opt/set_extra_paths.sh /opt/get_qwen_workflows.sh && \
    ln -s /opt/start_comfy.py /usr/local/bin/start-comfy && \
    ln -s /opt/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh

# Relax permissions for Toolbox non-root users
RUN chmod -R a+rwX /opt/ComfyUI

# Memory fragmentation mitigation ENV for PyTorch 2.0+
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Architecture fallback
ENV HSA_OVERRIDE_GFX_VERSION="9.0.0"

CMD ["/bin/bash"]
