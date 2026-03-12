# nvidia/cuda:12.9.1-devel-ubuntu24.04 is the default base
ARG DOCKER_REPO=reposmustbeset.azurecr.io

FROM ${DOCKER_REPO}/cuda:12.9.1-devel-ubuntu24.04

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
    conda run -n rtgen pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128 && \
    conda run -n rtgen pip install ninja packaging && \
    conda run -n rtgen pip cache purge

RUN conda run -n rtgen pip install git+https://github.com/huggingface/diffusers@22b229b && \
    conda run -n rtgen pip cache purge

COPY docker_files/base_requirements.txt /requirements.txt
RUN conda run -n rtgen pip install -r /requirements.txt && \
    conda run -n rtgen pip cache purge

# Install flash attention separately
RUN conda run -n rtgen pip install psutil numpy
RUN conda run -n rtgen pip install --no-build-isolation flash-attn==2.7.4.post1 && \
    conda run -n rtgen pip cache purge

# RUN git clone https://github.com/Dao-AILab/flash-attention && \
# RUN git clone --branch v2.7.4 https://github.com/Dao-AILab/flash-attention.git && \
# Skip flash-attn installation here but rather install it in requirements.txt
# RUN git clone https://github.com/Dao-AILab/flash-attention /tmp/flash-attention && \
#     cd /tmp/flash-attention && \
#     git checkout df1847a74ad0f9cee007ed186fab44f83fa03fad && \
#     cd hopper && \
#     conda run -n rtgen python setup.py install && \
#     rm -rf /tmp/flash-attention

RUN git clone https://github.com/Wan-Video/Wan2.1.git wan && \
    cd wan && \
    conda run -n rtgen pip install -r requirements.txt && \
    conda run -n rtgen pip cache purge

WORKDIR /wan

RUN conda run -n rtgen hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /wan/Wan2.1-I2V-14B-480P && \
    conda run -n rtgen hf download Wan-AI/Wan2.1-I2V-14B-720P --local-dir /wan/Wan2.1-I2V-14B-720P

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

COPY docker_files/vae.py /wan/wan/modules/vae.py

SHELL ["/bin/bash", "-c"]
RUN echo "source activate rtgen" >> ~/.bashrc

# docker run --gpus all -p 18080:8080 --name wan_rest wan
# docker exec -it wan_rest ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--wan", "--wan_variation", "480p"]
EXPOSE 8080
