# base image has CUDA, Miniconda, and streamwise conda env with torch
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

WORKDIR /streamwise

# Application dependencies
COPY requirements.txt .
RUN conda run -n streamwise pip install -r requirements.txt && \
    conda run -n streamwise pip cache purge

# Application files
COPY *.py .
COPY *.bash .
COPY *.sh .
COPY *.json .
COPY templates ./templates
COPY static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate streamwise" >> ~/.bashrc

ENTRYPOINT ["bash", "run_httpserver.bash", "--k8s_cluster", "incluster"]
EXPOSE 18181
