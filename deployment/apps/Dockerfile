# nvidia/cuda:12.9.1-devel-ubuntu24.04 is the default base
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG APP_NAME=streamwiseapp

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
    conda create -y -n streamwise python=3.12 && \
    conda install -n streamwise pip -y && \
    conda run -n streamwise pip install --upgrade pip && \
    conda run -n streamwise pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128 && \
    conda run -n streamwise pip install ninja packaging && \
    conda run -n streamwise pip cache purge

# LibreOffice to render slides
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libreoffice-core \
        libreoffice-impress \
        fonts-noto-color-emoji \
        fonts-noto-core \
        fonts-dejavu-core \
        fonts-noto-ui-core \
        fonts-liberation \
        ttf-mscorefonts-installer \
    && \
    fc-cache -f -v && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Main dependencies
COPY requirements_streamwise.txt .
RUN conda run -n streamwise pip install -r requirements_streamwise.txt

# Application specific
ARG APP_NAME
ENV APP_NAME=${APP_NAME}

WORKDIR /${APP_NAME}

COPY requirements_${APP_NAME}.txt .
RUN conda run -n streamwise pip install -r requirements_${APP_NAME}.txt && \
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

ENTRYPOINT bash run_httpserver.bash "$APP_NAME" --k8s_cluster incluster
EXPOSE 18080
