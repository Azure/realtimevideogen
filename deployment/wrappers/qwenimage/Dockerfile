# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

WORKDIR /qwenimage

# Download Qwen Image model from Hugging Face
RUN hf download Qwen/Qwen-Image --local-dir Qwen/Qwen-Image

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


# docker run --gpus all -p 18080:8080 --name qwenimage_rest qwenimage
# docker exec -it qwenimage_rest ls -l /
# TODO detect number of GPUs and set nproc_per_node accordingly
ENTRYPOINT ["bash", "run_httpserver.bash", "--qwenimage"]
EXPOSE 8080