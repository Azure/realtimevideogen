# base image has CUDA, Miniconda, streamwise conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

RUN git clone https://github.com/microsoft/VibeVoice.git vibevoice
WORKDIR /vibevoice

COPY docker_files/requirements.txt /vibevoice/requirements.txt
RUN conda run -n streamwise pip install -r /vibevoice/requirements.txt && \
    conda run -n streamwise pip cache purge

RUN conda run -n streamwise hf download microsoft/VibeVoice-1.5B --local-dir microsoft/VibeVoice-1.5B
RUN conda run -n streamwise hf download Qwen/Qwen2.5-1.5B --local-dir Qwen/Qwen2.5-1.5B

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static
COPY docker_files/services.json .

# VibeVoice files
COPY docker_files/schedule /vibevoice/schedule
COPY docker_files/voices /vibevoice/voices

SHELL ["/bin/bash", "-c"]
RUN echo "source activate streamwise" >> ~/.bashrc

# docker run --gpus all -p 18080:8080 vibevoice
# docker exec -it <container-id> ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--vibevoice"]
EXPOSE 8080