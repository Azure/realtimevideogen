ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_IMAGE=${DOCKER_REPO}/cuda:12.9.1-devel-ubuntu24.04

FROM ${BASE_IMAGE}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        wget \
        gnupg2 \
        git \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        libgl1 \
        ffmpeg \
        build-essential \
        cmake \
        mediainfo \
        python3 \
        python3-dev \
        python3-pip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG TORCH_PACKAGE=torch==2.8.0+cu128
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

RUN pip install "${TORCH_PACKAGE}" --index-url "${TORCH_INDEX_URL}" && \
    pip install ninja packaging && \
    pip cache purge

COPY base_requirements.txt /requirements.txt
RUN pip install -r /requirements.txt && \
    pip cache purge
