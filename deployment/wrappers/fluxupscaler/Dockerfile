# base image has CUDA, Miniconda, rtgen conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

RUN mkdir /fluxupscaler
WORKDIR /fluxupscaler

# Download FLUX.1-dev model from Hugging Face
ENV HF_HOME=/huggingface_hub
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) && \
    mkdir -p $HF_HOME && \
    conda run -n rtgen hf auth login --token "$HF_TOKEN" && \
    conda run -n rtgen hf download black-forest-labs/FLUX.1-dev --local-dir black-forest-labs/FLUX.1-dev
# Download ControlNet Upscaler model from Hugging Face
RUN conda run -n rtgen hf download jasperai/Flux.1-dev-Controlnet-Upscaler --local-dir jasperai/Flux.1-dev-Controlnet-Upscaler

COPY docker_files/requirements.txt .
RUN conda run -n rtgen pip install -r requirements.txt && \
    conda run -n rtgen pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate rtgen" >> ~/.bashrc

ENTRYPOINT ["bash", "run_httpserver.bash", "--fluxupscaler"]
EXPOSE 8080