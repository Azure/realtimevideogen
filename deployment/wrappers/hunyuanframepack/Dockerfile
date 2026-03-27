# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

RUN git clone https://github.com/lllyasviel/FramePack.git hunyuanframepack

# We avoid installing requirements.txt from the cloned repo to avoid diffusers overwrite
WORKDIR /hunyuanframepack

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

# Download HunyuanVideo and FramePackI2V_HY models from Hugging Face
RUN hf download hunyuanvideo-community/HunyuanVideo --local-dir hunyuanvideo-community/HunyuanVideo && \
    hf download lllyasviel/FramePackI2V_HY --local-dir lllyasviel/FramePackI2V_HY

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


# docker run --gpus all -p 18080:8080 hunyuanframepack
# docker exec -it <container-id> ls -l /
# TODO detect number of GPUs and set nproc_per_node accordingly
ENTRYPOINT ["bash", "run_httpserver.bash", "--hunyuanframepack"]
EXPOSE 8080
