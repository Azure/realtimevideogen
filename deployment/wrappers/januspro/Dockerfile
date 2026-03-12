# nvidia/cuda:12.9.1-devel-ubuntu24.04 is the default base
ARG DOCKER_REPO=reposmustbeset.azurecr.io

FROM ${DOCKER_REPO}/cuda:12.9.1-devel-ubuntu24.04

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
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n rtgen python=3.12 && \
    conda install -n rtgen pip -y && \
    conda run -n rtgen pip install --upgrade pip && \
    conda run -n rtgen pip install torch==2.7.1 ninja packaging && \
    conda run -n rtgen pip cache purge

RUN conda run -n rtgen pip install git+https://github.com/huggingface/diffusers@22b229b && \
    conda run -n rtgen pip cache purge

COPY docker_files/base_requirements.txt /requirements.txt
RUN conda run -n rtgen pip install -r /requirements.txt && \
    conda run -n rtgen pip cache purge

#RUN git clone https://github.com/Dao-AILab/flash-attention && \
#RUN git clone --branch v2.7.4 https://github.com/Dao-AILab/flash-attention.git && \
# Skip flash-attn installation here but rather install it in requirements.txt
# RUN git clone https://github.com/Dao-AILab/flash-attention /tmp/flash-attention && \
#     cd /tmp/flash-attention && \
#     git checkout df1847a74ad0f9cee007ed186fab44f83fa03fad && \
#     cd hopper && \
#     conda run -n rtgen python setup.py install && \
#     rm -rf /tmp/flash-attention

RUN git clone https://github.com/deepseek-ai/Janus.git januspro
WORKDIR /januspro

# Download januspro model from Hugging Face
RUN conda run -n rtgen hf download deepseek-ai/Janus-Pro-7B --local-dir deepseek-ai/Janus-Pro-7B

COPY docker_files/requirements.txt .
RUN conda run -n rtgen pip install -r requirements.txt && \
    conda run -n rtgen pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate rtgen" >> ~/.bashrc

# docker run --gpus all -p 18080:8080 --name januspro_rest januspro
# docker exec -it januspro_rest ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--januspro"]
EXPOSE 8080