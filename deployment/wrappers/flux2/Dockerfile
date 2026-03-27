# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

WORKDIR /flux2

# Download FLUX.2-dev model from Hugging Face
ENV HF_HOME=/huggingface_hub
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) && \
    mkdir -p $HF_HOME && \
    hf auth login --token "$HF_TOKEN" && \
    hf download black-forest-labs/FLUX.2-dev --local-dir black-forest-labs/FLUX.2-dev

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

COPY docker_files/*.py .
# Inject transformer_flux2.py (cloned from xDiT GitHub) into the installed xfuser package
RUN XFUSER_TRANSFORMERS=$(python3 -c \
        "import xfuser, os; print(os.path.join(os.path.dirname(os.path.abspath(xfuser.__file__)), \
        'model_executor/models/transformers'))") && \
    cp transformer_flux2.py "$XFUSER_TRANSFORMERS/transformer_flux2.py"
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


ENTRYPOINT ["bash", "run_httpserver.bash", "--flux2"]
EXPOSE 8080
