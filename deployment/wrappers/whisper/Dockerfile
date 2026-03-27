# Docker file extending vLLM image to add librosa for audio processing
# nvidia/cuda:12.9.1-devel-ubuntu24.04 is the default base
ARG DOCKER_REPO=reposmustbeset.azurecr.io

FROM ${DOCKER_REPO}/vllm/vllm-openai:v0.9.1

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}
ENV PIP_ROOT_USER_ACTION=ignore

# System deps (librosa needs soundfile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir librosa soundfile
