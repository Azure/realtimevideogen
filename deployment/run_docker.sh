#!/usr/bin/env bash

MAIN_DIR=$(realpath ..)
DEPLOYMENT_DIR=$(realpath .)
#shellcheck disable=SC1090,SC1091
source "$DEPLOYMENT_DIR/set_properties.sh"

# Parse arguments
SERVICE=$1
shift 1
PORT=18080
NUM_GPUS=2

# Get the rest of the arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$SERVICE" ]; then
    echo "Usage: $0 <service_name> --port <port> --gpus <num_gpus>"
    exit 1
fi

# Get the right directory
if [ -d "$DEPLOYMENT_DIR/wrappers/$SERVICE" ]; then
  SERVICE_DIR="$DEPLOYMENT_DIR/wrappers/$SERVICE"
elif [ -d "$DEPLOYMENT_DIR/apps/$SERVICE" ]; then
  SERVICE_DIR="$DEPLOYMENT_DIR/apps/$SERVICE"
else
  echo "Service directory for '$SERVICE' not found."
  exit 1
fi
cd "$SERVICE_DIR" || exit 1

# Build the image with the updated code
if [ "$(uname -m)" = "aarch64" ]; then
    PLATFORM_ARG="linux/arm64"
else
    PLATFORM_ARG="linux/amd64"
fi
bash setup_image.sh --platform "$PLATFORM_ARG"

# Check if container is running
if [ "$(docker ps -a -q -f name="${SERVICE}"_rest)" ]; then
    echo "Stopping existing container..."
    docker stop "${SERVICE}_rest"
    docker rm "${SERVICE}_rest"
fi

# Get the latest tag for the image
TAG=$(jq -r --arg name "$SERVICE" '.[$name].dockerImage.tag' "$MAIN_DIR/services.json")

# '"device=4,5"' or "device=4" or "device=7"
# --gpus '"device=0,1"' \
if ! docker run -d \
    --gpus "$NUM_GPUS" \
    -p "$PORT:8080" \
    --name "${SERVICE}_rest" \
    "${SERVICE}:${TAG}"; then
    echo "Failed to start the Docker container."
    exit 1
fi

# Access the service logs and status:
# curl http://localhost:18080/file/streamwise.log
# curl -s http://localhost:18080/health | jq .
watch --color -n 1 "curl -s http://localhost:$PORT/file/streamwise.log | tail -n 50"

# Live debugging:
# docker exec -it ${SERVICE}_rest cat /tmp/streamwise.log
# docker exec -it ${SERVICE}_rest bash
