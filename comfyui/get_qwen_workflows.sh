#!/usr/bin/env bash
# Downloads Qwen-Image and Qwen-Image-Edit GGUF models (Q4_K_M)
# plus the shared FP8 text encoder and VAE from Comfy-Org.
# All files land in ~/comfy-models/ which is mapped via extra_model_paths.yaml.
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODEL_HOME="$HOME/comfy-models"

# Detect hf CLI binary
HF=""
for candidate in hf huggingface-cli; do
  if command -v "$candidate" >/dev/null 2>&1; then
    HF="$candidate"
    break
  fi
done
if [[ -z "$HF" ]]; then
  echo "❌ Neither 'hf' nor 'huggingface-cli' found. Install huggingface_hub first."
  exit 1
fi

mkdir -p "$MODEL_HOME"/{diffusion_models,text_encoders,vae,loras}

dl() {
  local repo="$1" remote="$2" dest_dir="$3"
  local filename
  filename=$(basename "$remote")
  local dest="$dest_dir/$filename"

  if [[ -f "$dest" ]]; then
    echo "✓ Already present: $dest"
    return
  fi

  echo "↓ Downloading $filename from $repo..."
  "$HF" download "$repo" "$remote" \
      --repo-type model \
      --cache-dir "$HF_HOME" \
      --local-dir "$dest_dir"

  # HuggingFace CLI retains the remote folder structure (e.g. split_files/text_encoders/...)
  # We need to flatten it so ComfyUI finds them instantly inside the root category folder
  if [[ "$remote" == */* ]]; then
    find "$dest_dir" -type f -name "$filename" -exec mv {} "$dest_dir/" \; || true
    # Remove the empty downloaded folder structure
    local top_dir="${remote%%/*}"
    if [ -d "$dest_dir/$top_dir" ]; then
      rm -rf "$dest_dir/$top_dir"
    fi
  fi
}

echo ""
echo "============================================="
echo " Qwen-Image GGUF Downloader for MI50 (Q4_K_M)"
echo "============================================="
echo ""

# --- Shared components (text encoder + VAE) ---
echo ">>> [1/4] Text Encoder (FP8, ~7 GB)"
dl "Comfy-Org/Qwen-Image_ComfyUI" \
   "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
   "$MODEL_HOME/text_encoders"

echo ""
echo ">>> [2/4] VAE"
dl "Comfy-Org/Qwen-Image_ComfyUI" \
   "split_files/vae/qwen_image_vae.safetensors" \
   "$MODEL_HOME/vae"

# --- Qwen-Image 2512 (text-to-image) ---
echo ""
echo ">>> [3/6] Qwen-Image 2512 UNet (Q4_K_M GGUF, ~11.5 GB)"
dl "unsloth/Qwen-Image-2512-GGUF" \
   "qwen-image-2512-Q4_K_M.gguf" \
   "$MODEL_HOME/diffusion_models"

echo ""
echo ">>> [4/6] Qwen-Image 2512 Lightning LoRA (4-Step, ~3.5 GB)"
dl "lightx2v/Qwen-Image-2512-Lightning" \
   "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors" \
   "$MODEL_HOME/loras"

# --- Qwen-Image-Edit 2511 (image editing) ---
echo ""
echo ">>> [5/6] Qwen-Image-Edit 2511 UNet (Q4_K_M GGUF, ~11.5 GB)"
dl "unsloth/Qwen-Image-Edit-2511-GGUF" \
   "qwen-image-edit-2511-Q4_K_M.gguf" \
   "$MODEL_HOME/diffusion_models"

echo ""
echo ">>> [6/6] Qwen-Image-Edit 2511 Lightning LoRA (4-Step, ~3.5 GB)"
dl "lightx2v/Qwen-Image-Edit-2511-Lightning" \
   "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" \
   "$MODEL_HOME/loras"

echo ""
echo "============================================="
echo "✅ All Qwen-Image models downloaded!"
echo "   Location: $MODEL_HOME"
echo "============================================="
