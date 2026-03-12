# nvidia/cuda:12.9.1-devel-ubuntu24.04 is the default base
ARG DOCKER_REPO=reposmustbeset.azurecr.io

FROM ${DOCKER_REPO}/cuda:12.9.1-devel-ubuntu24.04

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}
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
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda clean -afy
# Python 3.12 -> 3.11 because:
# AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n rtgen python=3.11 && \
    conda install -n rtgen pip -y && \
    conda run -n rtgen pip install --upgrade pip && \
    conda run -n rtgen pip install torch==2.7.1 ninja packaging && \
    conda run -n rtgen pip cache purge

RUN conda run -n rtgen pip install git+https://github.com/huggingface/diffusers@22b229b && \
    conda run -n rtgen pip cache purge

COPY docker_files/base_requirements.txt /requirements.txt
RUN conda run -n rtgen pip install -r /requirements.txt && \
    conda run -n rtgen pip cache purge

# Get Hunyuan-Avatar repo and install dependencies
RUN git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git && \
    cd HunyuanVideo-Avatar && \
    mkdir -p /hunyuanavatar && \
    cp -r hymm_sp /hunyuanavatar/hymm_sp && \
    conda install -n rtgen pytorch-cuda -c pytorch -c nvidia && \
    conda run -n rtgen pip install -r requirements.txt && \
    # conda run -n rtgen pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.1 && \
    conda run -n rtgen pip cache purge
WORKDIR /hunyuanavatar

# Download models for Hunyuan-Avatar
RUN mkdir -p /hunyuanavatar/weights && \
    conda run -n rtgen hf download tencent/HunyuanVideo-Avatar --local-dir /hunyuanavatar/weights

# Prepare the working directory
COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

# COPY docker_files/config.py /hunyuanavatar/hymm_sp/config.py

SHELL ["/bin/bash", "-c"]
RUN echo "source activate rtgen" >> ~/.bashrc

ENV MODEL_BASE="/hunyuanavatar/weights"
ENTRYPOINT ["bash", "run_httpserver.bash", "--hunyuanavatar"]
EXPOSE 8080
