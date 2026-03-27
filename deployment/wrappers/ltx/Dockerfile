# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v0.5.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

WORKDIR /ltx

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

# Download LTX model from Hugging Face
RUN hf download Lightricks/LTX-Video-0.9.7-dev --local-dir Lightricks/LTX-Video-0.9.7-dev && \
    hf download Lightricks/ltxv-spatial-upscaler-0.9.7 --local-dir Lightricks/ltxv-spatial-upscaler-0.9.7

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


# docker run --gpus all -p 18080:8080 ltx
# docker exec -it <container-id> ls -l /
# TODO detect number of GPUs and set nproc_per_node accordingly
ENTRYPOINT ["bash", "run_httpserver.bash", "--ltx"]
EXPOSE 8080
