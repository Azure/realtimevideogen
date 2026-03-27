# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v0.5.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

WORKDIR /realesrgan

# RUN pip install git+https://github.com/ai-forever/Real-ESRGAN.git
RUN pip install setuptools wheel && \
    pip install --no-build-isolation git+https://github.com/ai-forever/Real-ESRGAN.git

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

# Download Real-ESRGAN model from Hugging Face
RUN huggingface-cli download ai-forever/Real-ESRGAN --local-dir ai-forever/Real-ESRGAN

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


ENTRYPOINT ["bash", "run_httpserver.bash", "--realesrgan"]
EXPOSE 8080
