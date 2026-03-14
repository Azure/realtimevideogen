#!/usr/bin/env bash

NUM_GPUS=0 # if nvidia-smi is not available, we assume no GPUs are present
if command -v nvidia-smi &> /dev/null; then
  NUM_GPUS=$(nvidia-smi -L | wc -l)

  echo "GPUs:"
  nvidia-smi -L
  nvidia-smi
fi

echo "Number of GPUs detected: ${NUM_GPUS}"

echo ""
echo "Storage:"
df -h

echo ""
echo "Memory:"
free -h

echo "CPU:"
lscpu

echo ""
echo "Environment variables:"
printenv

echo "FFmpeg:"
ffmpeg -version
dpkg -s ffmpeg

# Allow live logging
export PYTHONUNBUFFERED=1
# A100: 8.0
# H100: 9.0
export TORCH_CUDA_ARCH_LIST="8.0 9.0"

if [[ $NUM_GPUS -gt 8 ]]; then
    # Multiple server setup
    # TODO we need to get total number of GPUs across all nodes
    GPUS_PER_SERVER=$(nvidia-smi -L | wc -l)
    NUM_SERVERS=$((NUM_GPUS / GPUS_PER_SERVER))
    MAIN_SERVER=10.0.0.4:29500  # Set this up
    NODE_RANK=0  # Set this up

    /opt/conda/bin/conda run -n streamwise \
    torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MAIN_SERVER}" \
    --nnodes="${NUM_SERVERS}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${GPUS_PER_SERVER}" \
    run_httpserver.py \
    --ulysses_degree "${GPUS_PER_SERVER}" \
    --ring_degree "${NUM_SERVERS}" \
    --use_torch_compile \
    "$@"
elif [[ $NUM_GPUS -gt 1 ]]; then
    # Single-node multi-GPU setup
    /opt/conda/bin/conda run -n streamwise \
    torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    run_httpserver.py \
    --ulysses_degree "${NUM_GPUS}" \
    --ring_degree 1 \
    --use_torch_compile \
    "$@"
else
    /opt/conda/bin/conda run -n streamwise \
    python3 -u run_httpserver.py \
    "$@"
fi
