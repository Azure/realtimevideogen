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
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n streamwise python=3.12 && \
    conda install -n streamwise pip -y && \
    conda run -n streamwise pip install --upgrade pip && \
    conda run -n streamwise pip install torch==2.7.1 ninja packaging && \
    conda run -n streamwise pip cache purge

WORKDIR /streamwise

# Application dependencies
COPY requirements.txt .
RUN conda run -n streamwise pip install -r requirements.txt && \
    conda run -n streamwise pip cache purge

# Application files
COPY *.py .
COPY *.bash .
COPY *.sh .
COPY templates ./templates
COPY static ./static

COPY services.json .

SHELL ["/bin/bash", "-c"]
RUN echo "source activate streamwise" >> ~/.bashrc

ENTRYPOINT ["bash", "run_httpserver.bash", "--k8s_cluster", "incluster"]
EXPOSE 18181
