ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_IMAGE=${DOCKER_REPO}/cuda:12.9.1-devel-ubuntu24.04

FROM ${BASE_IMAGE}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}
ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        wget \
        gnupg2 \
        git \
        bzip2 \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        libgl1 \
        ffmpeg \
        build-essential \
        cmake \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        echo "Unsupported architecture: $TARGETARCH" && exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda clean -afy

ARG TORCH_PACKAGE=torch==2.8.0+cu128
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n streamwise python=3.12 && \
    conda install -n streamwise pip -y && \
    conda run -n streamwise pip install --upgrade pip && \
    conda run -n streamwise pip install "${TORCH_PACKAGE}" --index-url "${TORCH_INDEX_URL}" && \
    conda run -n streamwise pip install ninja packaging && \
    conda run -n streamwise pip cache purge

COPY base_requirements.txt /requirements.txt
RUN conda run -n streamwise pip install -r /requirements.txt && \
    conda run -n streamwise pip cache purge
