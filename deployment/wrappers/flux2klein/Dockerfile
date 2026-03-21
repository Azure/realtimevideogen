# base image has CUDA, Miniconda, streamwise conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

WORKDIR /flux2klein

# Download FLUX.2-klein-9B model from Hugging Face
ENV HF_HOME=/huggingface_hub
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) && \
    mkdir -p $HF_HOME && \
    conda run -n streamwise hf auth login --token "$HF_TOKEN" && \
    conda run -n streamwise hf download black-forest-labs/FLUX.2-klein-9B --local-dir black-forest-labs/FLUX.2-klein-9B

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

ENTRYPOINT ["bash", "run_httpserver.bash", "--flux2klein"]
EXPOSE 8080
