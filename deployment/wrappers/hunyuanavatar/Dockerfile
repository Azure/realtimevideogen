# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v0.5.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

# Get Hunyuan-Avatar repo and install dependencies
RUN git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git /hunyuanavatar
WORKDIR /hunyuanavatar

# Use our patched requirements to work around pins incompatible with the current Python 3.12/base image
COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

# Download models for Hunyuan-Avatar
RUN mkdir -p /hunyuanavatar/weights && \
    hf download tencent/HunyuanVideo-Avatar --local-dir /hunyuanavatar/weights

# Prepare the working directory
COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

# COPY docker_files/config.py /hunyuanavatar/hymm_sp/config.py


ENV MODEL_BASE="/hunyuanavatar/weights"
ENTRYPOINT ["bash", "run_httpserver.bash", "--hunyuanavatar"]
EXPOSE 8080
