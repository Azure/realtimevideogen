#!/usr/bin/env bash

# shellcheck disable=SC1091
source ../../set_properties.sh

docker buildx build \
  --build-arg DOCKER_REPO="${ACR_URL}" \
  -t "$ACR_URL/vllm-librosa:v0.9.1" \
  .
docker push "$ACR_URL/vllm-librosa:v0.9.1"
