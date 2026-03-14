# base image has CUDA, Miniconda, rtgen conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

RUN git clone https://github.com/FunAudioLLM/ThinkSound.git thinksound
WORKDIR /thinksound

# Download thinksound model from Hugging Face
RUN conda run -n rtgen hf download FunAudioLLM/ThinkSound --local-dir FunAudioLLM/ThinkSound

COPY docker_files/requirements.txt .
RUN conda run -n rtgen pip install -r requirements.txt && \
    conda run -n rtgen pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate rtgen" >> ~/.bashrc

# docker run --gpus all -p 18080:8080 --name thinksound_rest thinksound
# docker exec -it thinksound_rest ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--thinksound"]
EXPOSE 8080