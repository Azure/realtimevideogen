# base image has CUDA, Miniconda, rtgen conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

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

COPY docker_files/base_requirements.txt /requirements.txt
RUN conda run -n rtgen pip install -r /requirements.txt && \
    conda run -n rtgen pip cache purge

WORKDIR /slidetranscript

COPY docker_files/requirements.txt .
RUN conda run -n rtgen pip install -r requirements.txt && \
    conda run -n rtgen pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate rtgen" >> ~/.bashrc

ENTRYPOINT ["bash", "run_httpserver.bash", "--slidetranscript"]
EXPOSE 8080