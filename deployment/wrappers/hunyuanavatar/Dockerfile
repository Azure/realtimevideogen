# base image has CUDA, Miniconda, streamwise conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

# Get Hunyuan-Avatar repo and install dependencies
RUN git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git && \
    cd HunyuanVideo-Avatar && \
    mkdir -p /hunyuanavatar && \
    cp -r hymm_sp /hunyuanavatar/hymm_sp && \
    conda install -n streamwise pytorch-cuda -c pytorch -c nvidia && \
    conda run -n streamwise pip install -r requirements.txt && \
    # conda run -n streamwise pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.1 && \
    conda run -n streamwise pip cache purge
WORKDIR /hunyuanavatar

# Download models for Hunyuan-Avatar
RUN mkdir -p /hunyuanavatar/weights && \
    conda run -n streamwise hf download tencent/HunyuanVideo-Avatar --local-dir /hunyuanavatar/weights

# Prepare the working directory
COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

# COPY docker_files/config.py /hunyuanavatar/hymm_sp/config.py

SHELL ["/bin/bash", "-c"]
RUN echo "source activate streamwise" >> ~/.bashrc

ENV MODEL_BASE="/hunyuanavatar/weights"
ENTRYPOINT ["bash", "run_httpserver.bash", "--hunyuanavatar"]
EXPOSE 8080
