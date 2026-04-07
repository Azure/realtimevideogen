# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v0.5.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

# Install flash attention separately
RUN pip install psutil numpy
RUN pip install --no-build-isolation flash-attn==2.8.2 && \
    pip cache purge

RUN git clone https://github.com/microsoft/VibeVoice.git vibevoice
WORKDIR /vibevoice

COPY docker_files/requirements.txt /vibevoice/requirements.txt
RUN pip install -r /vibevoice/requirements.txt && \
    pip cache purge

RUN hf download microsoft/VibeVoice-1.5B --local-dir microsoft/VibeVoice-1.5B
RUN hf download Qwen/Qwen2.5-1.5B --local-dir Qwen/Qwen2.5-1.5B

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

# VibeVoice files
COPY docker_files/schedule /vibevoice/schedule
COPY docker_files/voices /vibevoice/voices


# docker run --gpus all -p 18080:8080 vibevoice
# docker exec -it <container-id> ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--vibevoice"]
EXPOSE 8080