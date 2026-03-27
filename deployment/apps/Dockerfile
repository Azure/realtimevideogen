# base image has CUDA and Python with torch
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG APP_NAME=streamwiseapp
ARG BASE_TAG=v0.5.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

# LibreOffice to render slides
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libreoffice-core \
        libreoffice-impress \
        fonts-noto-color-emoji \
        fonts-noto-core \
        fonts-dejavu-core \
        fonts-noto-ui-core \
        fonts-liberation \
        ttf-mscorefonts-installer \
    && \
    fc-cache -f -v && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Main dependencies
COPY requirements_streamwise.txt .
RUN pip install -r requirements_streamwise.txt

# Application specific
ARG APP_NAME
ENV APP_NAME=${APP_NAME}

WORKDIR /${APP_NAME}

COPY requirements_${APP_NAME}.txt .
RUN pip install -r requirements_${APP_NAME}.txt && \
    pip cache purge

# Application files
COPY *.py .
COPY *.bash .
COPY *.sh .
COPY *.json .
COPY templates ./templates
COPY static ./static

# TLS certificates (optional — populated by setup_image.sh --certfile/--keyfile or mounted at runtime)
RUN mkdir -p /certs
COPY certs /certs

ENTRYPOINT bash run_httpserver.bash "$APP_NAME" --k8s_cluster incluster
EXPOSE 18080
