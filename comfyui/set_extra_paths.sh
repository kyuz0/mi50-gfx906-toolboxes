#!/bin/bash
# Links ComfyUI's internal model paths natively to your host machine's home directory.

set -euo pipefail

COMFY_DIR="/opt/ComfyUI"
YAML_FILE="$COMFY_DIR/extra_model_paths.yaml"
MODEL_DIR="$HOME/comfy-models"

mkdir -p "$MODEL_DIR"/{text_encoders,vae,diffusion_models,loras,unet,checkpoints}

cat > "$YAML_FILE" <<EOF
comfyui:
    base_path: $MODEL_DIR

    checkpoints: checkpoints
    text_encoders: text_encoders
    clip: text_encoders
    vae: vae
    diffusion_models: diffusion_models
    unet: unet
    loras: loras
    latent_upscale_models: latent_upscale_models
    clip_vision: clip_vision
EOF

echo "============================================="
echo "✅ Default host volume mapping deployed!"
echo "Mapped host directory: $MODEL_DIR"
echo "Target YAML mapping:   $YAML_FILE"
echo "============================================="
