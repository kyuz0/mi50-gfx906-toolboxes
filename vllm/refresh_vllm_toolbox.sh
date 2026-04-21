#!/usr/bin/env bash
set -e

TOOLBOX_NAME="vllm-gfx906-7.2.1"
IMAGE="docker.io/kyuz0/vllm-toolbox-gfx906:0.19.1-rocm-7.2.1-aiinfos"

# Get the current image ID BEFORE pulling
OLD_IMAGE_ID=$(podman inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo "")

echo ">>> Pulling latest image from registry: $IMAGE"
podman pull "$IMAGE"

# Get the latest image ID AFTER pulling
IMAGE_ID=$(podman inspect -f '{{.Id}}' "$IMAGE")

# Check if the toolbox container exists
if podman container exists "$TOOLBOX_NAME"; then
    echo ">>> Toolbox '$TOOLBOX_NAME' exists. Checking version..."
    CONTAINER_IMAGE_ID=$(podman inspect -f '{{.Image}}' "$TOOLBOX_NAME")
    
    if [ "$IMAGE_ID" == "$CONTAINER_IMAGE_ID" ]; then
        echo ">>> Toolbox is already up to date!"
        echo ">>> Run 'toolbox enter $TOOLBOX_NAME' to start."
        exit 0
    else
        echo ">>> New image build detected! Removing old toolbox..."
        # If it's running, toolbox will stop it on rm -f
        podman rm -f "$TOOLBOX_NAME"
    fi
else
    echo ">>> Toolbox '$TOOLBOX_NAME' not found. Will create a fresh one."
fi

echo ">>> Creating new toolbox..."
toolbox create "$TOOLBOX_NAME" \
    --image "$IMAGE" \
    -- \
    --device /dev/dri --device /dev/kfd \
    --group-add video --group-add render --group-add sudo \
    --security-opt seccomp=unconfined

# Surgically remove ONLY the specific old vLLM image if it was replaced
if [ -n "$OLD_IMAGE_ID" ] && [ "$OLD_IMAGE_ID" != "$IMAGE_ID" ]; then
    echo ">>> Cleaning up previous vLLM toolbox image ($OLD_IMAGE_ID)..."
    podman rmi -f "$OLD_IMAGE_ID" || true
fi

echo "============================================="
echo "✅ Toolbox successfully refreshed and ready!"
echo "👉 Enter with: toolbox enter $TOOLBOX_NAME"
echo "============================================="
