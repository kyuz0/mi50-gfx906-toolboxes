#!/usr/bin/env bash
set -e

refresh_toolbox() {
    local ROCM_VERSION=$1
    # Convert dots to hyphens to match standard toolbox naming constraints better
    local TOOLBOX_NAME="llama-gfx906-rocm${ROCM_VERSION//./-}"
    local IMAGE="docker.io/kyuz0/llamacpp-toolbox-gfx906:rocm-${ROCM_VERSION}"

    echo "============================================="
    echo ">>> Refreshing Toolbox: $TOOLBOX_NAME "
    echo "============================================="

    # Get the current image ID BEFORE pulling
    local OLD_IMAGE_ID=$(podman inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo "")

    echo ">>> Pulling latest image from registry: $IMAGE"
    podman pull "$IMAGE" || echo "Warning: Could not pull. Using local cache if it exists."

    # Get the latest image ID AFTER pulling
    local IMAGE_ID=$(podman inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo "")
    
    if [ -z "$IMAGE_ID" ]; then
        echo "❌ Error: Could not resolve image ID for $IMAGE. Make sure it exists locally or on Docker Hub!"
        return 1
    fi

    # Check if the toolbox container exists
    if podman container exists "$TOOLBOX_NAME"; then
        echo ">>> Toolbox '$TOOLBOX_NAME' exists. Checking version..."
        local CONTAINER_IMAGE_ID=$(podman inspect -f '{{.Image}}' "$TOOLBOX_NAME")
        
        if [ "$IMAGE_ID" == "$CONTAINER_IMAGE_ID" ]; then
            echo ">>> Toolbox is already up to date!"
            echo ">>> Run 'toolbox enter $TOOLBOX_NAME' to start."
            return 0
        else
            echo ">>> New image build detected! Removing old toolbox..."
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

    # Surgically remove ONLY the specific old image if it was replaced
    if [ -n "$OLD_IMAGE_ID" ] && [ "$OLD_IMAGE_ID" != "$IMAGE_ID" ]; then
        echo ">>> Cleaning up previous image ($OLD_IMAGE_ID)..."
        podman rmi -f "$OLD_IMAGE_ID" || true
    fi

    echo "✅ Toolbox $TOOLBOX_NAME successfully refreshed!"
    echo ""
}

show_menu() {
    echo "Which llama.cpp toolbox would you like to refresh?"
    echo "1) ROCm 6.4.4 (docker.io/kyuz0/llamacpp-toolbox-gfx906:rocm-6.4.4)"
    echo "2) ROCm 7.2.1 (docker.io/kyuz0/llamacpp-toolbox-gfx906:rocm-7.2.1)"
    echo "3) All (Both)"
    echo "4) Cancel"
    read -p "Select an option [1-4]: " choice

    case $choice in
        1)
            refresh_toolbox "6.4.4"
            ;;
        2)
            refresh_toolbox "7.2.1"
            ;;
        3)
            refresh_toolbox "6.4.4"
            refresh_toolbox "7.2.1"
            ;;
        4)
            echo "Canceled."
            exit 0
            ;;
        *)
            echo "Invalid selection."
            exit 1
            ;;
    esac
}

# If arguments are passed, use them instead of the menu
if [ "$1" == "6.4.4" ]; then
    refresh_toolbox "6.4.4"
elif [ "$1" == "7.2.1" ]; then
    refresh_toolbox "7.2.1"
elif [ "$1" == "all" ]; then
    refresh_toolbox "6.4.4"
    refresh_toolbox "7.2.1"
else
    show_menu
fi

echo "============================================="
echo "🎉 Done! Your llama.cpp toolboxes are ready."
echo "============================================="
