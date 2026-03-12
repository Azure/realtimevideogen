#!/usr/bin/env bash

# Start the vLLM server in a Docker container
export NUM_GPUS=4
export NUM_GPUS=1
export HF_HOME="/mnt/raid0/huggingface/"
export PORT=18088
export NETWORK_NAME="myvnet"
export HF_TOKEN="hf_XXXX"  # TODO fill in your Hugging Face token
export CONTAINER_NAME="vllm"
export MODEL="google/gemma-3-27b-it"
export VLLM_IMAGE="vllm/vllm-openai:latest"

# GB200
export HF_HOME="/data/huggingface/"
export VLLM_IMAGE="vllm/vllm-openai:cu130-nightly-aarch64"


if ! docker network inspect $NETWORK_NAME >/dev/null 2>&1; then
    docker network create $NETWORK_NAME
fi
docker run \
  --runtime nvidia \
  --name $CONTAINER_NAME \
  --gpus "$NUM_GPUS" \
  -v "$HF_HOME":/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  --network $NETWORK_NAME \
  -p $PORT:8000 \
  --ipc=host \
  $VLLM_IMAGE \
  --model $MODEL \
  --tensor-parallel-size "$NUM_GPUS" \
  --no-enable-prefix-caching


# From inside the Docker container
docker exec -it $CONTAINER_NAME /bin/bash
# Check the server is running
curl http://127.0.0.1:8000/v1/models


# Random dataset
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --dataset-name random \
  --model $MODEL


# ShareGPT dataset
mkdir -p /data/sharegpt
cd /data/sharegpt || exit
# apt update && apt install wget jq -y
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --dataset-name sharegpt \
  --dataset-path /data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model $MODEL


# Benchmark for StreamCast
# Mean TTFT -> first_scene_time
# Mean TBT x 6 (avg token per scene) -> per_scene_time
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --dataset-name random \
  --model $MODEL \
  --random-input-len=4096 \
  --random-output-len=60 \
  --num_prompts=10
