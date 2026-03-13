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

RUN git clone https://github.com/FoundationVision/LlamaGen llamagen

WORKDIR /llamagen

# Download LlamaGen model from Hugging Face
RUN conda run -n rtgen huggingface-cli download FoundationVision/LlamaGen --local-dir FoundationVision/LlamaGen
#RUN conda run -n rtgen huggingface-cli download DeepFloyd/t5-v1_1-xxl --local-dir DeepFloyd/t5-v1_1-xxl
RUN conda run -n rtgen huggingface-cli download google/flan-t5-xl --local-dir google/flan-t5-xl \
    config.json \
    special_tokens_map.json \
    spiece.model \
    tokenizer_config.json \
    pytorch_model.bin.index.json \
    pytorch_model-00001-of-00002.bin \
    pytorch_model-00002-of-00002.bin
RUN conda run -n rtgen huggingface-cli download peizesun/llamagen_t2i --local-dir peizesun/llamagen_t2i

COPY docker_files/requirements.txt .
RUN conda run -n rtgen pip install -r requirements.txt && \
    conda run -n rtgen pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate rtgen" >> ~/.bashrc

# docker run --gpus all -p 18080:8080 --name llamagen_rest llamagen
# docker exec -it llamagen_rest ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--llamagen"]
EXPOSE 8080