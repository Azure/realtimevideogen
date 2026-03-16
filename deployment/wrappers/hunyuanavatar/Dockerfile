# base image has CUDA, Miniconda, streamwise conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

# Get Hunyuan-Avatar repo and install dependencies
RUN git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git /hunyuanavatar && \
    conda install -n streamwise pytorch-cuda -c pytorch -c nvidia
WORKDIR /hunyuanavatar

# Use our patched requirements instead of the ones in the repo for old Python versions
COPY docker_files/requirements.txt .
RUN conda run -n streamwise pip install -r requirements.txt && \
    conda run -n streamwise pip cache purge

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
