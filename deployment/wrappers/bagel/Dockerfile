# base image has CUDA, Miniconda, streamwise conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

RUN git clone https://github.com/ByteDance-Seed/Bagel.git bagel

WORKDIR /bagel

COPY docker_files/requirements.txt /bagel/requirements.txt
RUN conda run -n streamwise pip install -r /bagel/requirements.txt && \
    conda run -n streamwise pip cache purge

# Download BAGEL-7B-MoT model from Hugging Face
RUN conda run -n streamwise hf download ByteDance-Seed/BAGEL-7B-MoT --local-dir /bagel/BAGEL-7B-MoT

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate streamwise" >> ~/.bashrc

ENTRYPOINT ["bash", "run_httpserver.bash", "--bagel"]
EXPOSE 8080
