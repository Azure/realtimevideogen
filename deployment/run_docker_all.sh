#!/usr/bin/env bash

# Deploy all the main docker images locally.

NETWORK_NAME="mynet"

docker network create "$NETWORK_NAME"

# Documents to podcast transcript
docker run -d --gpus '"device=0"' \
    --network "$NETWORK_NAME" \
    -p 18080:8080 \
    --name podcasttranscript_rest podcasttranscript

# Text to audio
docker run -d --gpus '"device=0"' -p 18081:8080 --name kokoro_rest kokoro

# Image to video
docker run -d --gpus '"device=2,3"' -p 18082:8080 --name hunyuanframepack_rest hunyuanframepack
docker run -d --gpus '"device=4,5"' -p 18083:8080 --name wan_rest wan

# Image+audio to video
docker run -d --gpus '"device=6,7"' -p 18084:8080 --name fantasytalking_rest fantasytalking

# Text to image
docker run -d --gpus '"device=6,7"' -p 18085:8080 --name flux_rest flux

# Image recognition
docker run -d --gpus '"device=0"' -p 18088:8080 --name yolo_rest yolo

# Image editing
docker run -d --gpus '"device=0"' -p 18089:8080 --name bagel_rest bagel

# Check environment variables
if [[ -z "${HF_HOME:-}" ]]; then
    echo "ERROR: HF_HOME environment variable must be set"
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN environment variable must be set"
    exit 1
fi

# vLLM with Gemma model
docker run -d \
  --gpus '"device=0,1"' \
  -v "$HF_HOME:/root/.cache/huggingface" \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  --network "$NETWORK_NAME" \
  --name gemma \
  -p 18086:8000 \
  --ipc=host \
  vllm/vllm-openai:v0.8.4 \
  --model google/gemma-3-27b-it \
  --tensor-parallel-size 2 \
  --guided-decoding-backend xgrammar
