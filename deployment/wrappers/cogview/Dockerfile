# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v0.5.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

WORKDIR /cogview

# Download CogView4 model from Hugging Face
RUN hf download THUDM/CogView4-6B --local-dir THUDM/CogView4-6B

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


# docker run --gpus all -p 18080:8080 --name cogview_rest cogview
# docker exec -it cogview_rest ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--cogview"]
EXPOSE 8080
