# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

RUN git clone https://github.com/coqui-ai/TTS.git xtts
# Use our own version instead
# pip install -r requirements.txt

WORKDIR /xtts

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

RUN hf download coqui/XTTS-v2 --local-dir coqui/XTTS-v2

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


# docker run --gpus all -p 18080:8080 xtts
# docker exec -it <container-id> ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--xtts"]
EXPOSE 8080
