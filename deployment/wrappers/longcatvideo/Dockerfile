# base image has CUDA, Miniconda, rtgen conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

WORKDIR /longcatvideo

# Downloadmodel from Hugging Face
# TODO

COPY docker_files/requirements.txt .
RUN conda run -n rtgen pip install -r requirements.txt && \
    conda run -n rtgen pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate rtgen" >> ~/.bashrc

# docker run --gpus all -p 18080:8080 --name longcatvideo_rest longcatvideo
# docker exec -it longcatvideo_rest ls -l /
# TODO detect number of GPUs and set nproc_per_node accordingly
ENTRYPOINT ["bash", "run_httpserver.bash", "--longcatvideo"]
EXPOSE 8080