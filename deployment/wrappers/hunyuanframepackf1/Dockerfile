# nvidia/cuda:12.9.1-devel-ubuntu24.04 is the default base
ARG DOCKER_REPO=reposmustbeset.azurecr.io

FROM ${DOCKER_REPO}/cuda:12.9.1-devel-ubuntu24.04

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
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n rtgen python=3.12 && \
    conda install -n rtgen pip -y && \
    conda run -n rtgen pip install --upgrade pip && \
    conda run -n rtgen pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128 && \
    conda run -n rtgen pip install ninja packaging && \
    conda run -n rtgen pip cache purge

COPY docker_files/base_requirements.txt /requirements.txt
RUN conda run -n rtgen pip install -r /requirements.txt && \
    conda run -n rtgen pip cache purge

# Install flash attention separately
RUN conda run -n rtgen pip install psutil numpy
RUN conda run -n rtgen pip install --no-build-isolation flash-attn==2.8.2 && \
    conda run -n rtgen pip cache purge

RUN git clone https://github.com/lllyasviel/FramePack.git hunyuanframepackf1

# We avoid installing requirements.txt from the cloned repo to avoid diffusers overwrite
WORKDIR /hunyuanframepackf1

COPY docker_files/requirements.txt .
RUN conda run -n rtgen pip install -r requirements.txt && \
    conda run -n rtgen pip cache purge

# Download HunyuanVideo and FramePackI2V_HY models from Hugging Face
RUN conda run -n rtgen hf download hunyuanvideo-community/HunyuanVideo --local-dir hunyuanvideo-community/HunyuanVideo && \
    conda run -n rtgen hf download lllyasviel/FramePack_F1_I2V_HY_20250503 --local-dir lllyasviel/FramePack_F1_I2V_HY_20250503

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate rtgen" >> ~/.bashrc

ENTRYPOINT ["bash", "run_httpserver.bash", "--hunyuanframepackf1"]
EXPOSE 8080
