# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

RUN git clone https://github.com/deepseek-ai/Janus.git januspro
WORKDIR /januspro

# Download januspro model from Hugging Face
RUN hf download deepseek-ai/Janus-Pro-7B --local-dir deepseek-ai/Janus-Pro-7B

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


# docker run --gpus all -p 18080:8080 --name januspro_rest januspro
# docker exec -it januspro_rest ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--januspro"]
EXPOSE 8080