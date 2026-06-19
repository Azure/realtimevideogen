# base image has CUDA and Python with torch
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v0.5.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

WORKDIR /streamwise

# Application dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

# Application files
COPY *.py .
COPY *.bash .
COPY *.sh .
COPY *.json .
COPY templates ./templates
COPY static ./static
COPY model_provisioner ./model_provisioner
COPY simulator ./simulator

# TLS certificates (optional — populated by setup_image.sh --certfile/--keyfile or mounted at runtime)
RUN mkdir -p /certs
COPY certs /certs

ENTRYPOINT ["bash", "run_httpserver.bash", "--k8s_cluster", "incluster"]
EXPOSE 18181
