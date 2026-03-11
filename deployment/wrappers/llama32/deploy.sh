#!/usr/bin/env bash

NUM_GPUS=2
PORT=18086

DEPLOYMENT_DIR="../.."
# shellcheck disable=SC1090,SC1091
source "$DEPLOYMENT_DIR/set_properties.sh"

while [[ $# -gt 0 ]]; do
    case $1 in
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

if [[ -z "${HF_HOME:-}" ]]; then
    echo "ERROR: HF_HOME environment variable must be set"
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN environment variable must be set"
    exit 1
fi

NETWORK_NAME="mynet"
if ! docker network inspect $NETWORK_NAME >/dev/null 2>&1; then
    docker network create $NETWORK_NAME
fi

docker run \
  --runtime nvidia \
  --gpus "$NUM_GPUS" \
  -v "$HF_HOME":/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  --network $NETWORK_NAME \
  -p $PORT:8000 \
  --ipc=host \
  "$DOCKER_REPO/vllm/vllm-openai:v0.9.1" \
  --model meta-llama/Llama-3.2-90B-Vision \
  --tensor-parallel-size "$NUM_GPUS" \
  --no-enable-prefix-caching \
  --guided-decoding-backend xgrammar
