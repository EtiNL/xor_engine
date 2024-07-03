#!/bin/bash

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq could not be found. Please install jq."
    exit 1
fi

# Parse devcontainer.json to get build context and Dockerfile
CONTEXT=$(jq -r '.build.context' .devcontainer/devcontainer.json)
DOCKERFILE=$(jq -r '.build.dockerfile' .devcontainer/devcontainer.json)
IMAGE_NAME="rust-sdl2-devcontainer"

# Build the Docker image
docker build -t $IMAGE_NAME -f .devcontainer/$DOCKERFILE $CONTEXT

# Parse runArgs
RUN_ARGS=$(jq -r '.runArgs | join(" ")' .devcontainer/devcontainer.json)

# Parse workspace folder, defaulting to /app if not set
WORKSPACE_FOLDER=$(jq -r '.workspaceFolder // "/app"' .devcontainer/devcontainer.json)

# Run the Docker container with parsed arguments
docker run -it --rm \
    $RUN_ARGS \
    -v "$(pwd)":$WORKSPACE_FOLDER \
    -w $WORKSPACE_FOLDER \
    $IMAGE_NAME \
    /bin/bash -c "cargo build && /bin/bash"
