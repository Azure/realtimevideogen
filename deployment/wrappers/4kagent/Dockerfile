# base image has CUDA, Miniconda, streamwise conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

# Install system libraries required to compile Pillow from source (e.g. libjpeg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libjpeg-dev \
        zlib1g-dev \
        libtiff-dev \
        libfreetype-dev \
        liblcms2-dev \
        libwebp-dev \
        libopenjp2-7-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone 4KAgent
RUN git clone --depth 1 https://github.com/taco-group/4KAgent /4kagent/4KAgent

WORKDIR /4kagent

# Install 4KAgent dependencies into the streamwise environment
RUN conda run -n streamwise pip install -r /4kagent/4KAgent/installation/requirements.txt && \
    conda run -n streamwise pip cache purge

# Create tool symbolic links
RUN cd /4kagent/4KAgent && bash installation/tools_sym_links.sh

# Download pretrained checkpoints from Hugging Face and extract the tarball
RUN conda run -n streamwise hf download YSZuo/4KAgent-Toolbox-Pretrained-Models \
        --local-dir /4kagent/4KAgent/pretrained_ckpts && \
    tar -xzf /4kagent/4KAgent/pretrained_ckpts/4KAgent_toolbox_pretrained_ckpts.tar.gz \
        -C /4kagent/4KAgent/pretrained_ckpts && \
    cp /4kagent/4KAgent/pretrained_ckpts/BOBL/Face_Detection/shape_predictor_68_face_landmarks.dat \
       /4kagent/4KAgent/executor/old_photo_restoration/tools/BOBL/Face_Detection/ && \
    cp /4kagent/4KAgent/pretrained_ckpts/hpsv2/bpe_simple_vocab_16e6.txt.gz \
       /4kagent/4KAgent/utils/clib_fiqa/model/

COPY docker_files/requirements.txt .
RUN conda run -n streamwise pip install -r requirements.txt && \
    conda run -n streamwise pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate streamwise" >> ~/.bashrc

ENTRYPOINT ["bash", "run_httpserver.bash", "--4kagent"]
EXPOSE 8080
