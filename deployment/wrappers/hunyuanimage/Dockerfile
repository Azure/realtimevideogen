# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

WORKDIR /hunyuanimage

# Download Qwen Image model from Hugging Face
# RUN hf download tencent/HunyuanImage-3.0 --local-dir tencent/HunyuanImage-3.0
RUN hf download tencent/HunyuanImage-3.0 --local-dir ./HunyuanImage-3

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


ENTRYPOINT ["bash", "run_httpserver.bash", "--hunyuanimage"]
EXPOSE 8080