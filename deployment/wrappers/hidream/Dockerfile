# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v0.5.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

WORKDIR /hidream

# Download HiDream model from Hugging Face
RUN hf download HiDream-ai/HiDream-I1-Full --local-dir HiDream-ai/HiDream-I1-Full
# Download Llama models from Hugging Face
ENV HF_HOME=/huggingface_hub
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) && \
    mkdir -p $HF_HOME && \
    hf auth login --token "$HF_TOKEN" && \
    hf download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir meta-llama/Meta-Llama-3.1-8B-Instruct

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


ENTRYPOINT ["bash", "run_httpserver.bash", "--hidream"]
EXPOSE 8080