# base image has CUDA, Miniconda, streamwise conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

RUN git clone https://github.com/nari-labs/dia.git

WORKDIR /dia

COPY docker_files/requirements.txt .
RUN conda run -n streamwise pip install -r requirements.txt && \
    conda run -n streamwise pip cache purge

RUN conda run -n streamwise hf download nari-labs/Dia-1.6B-0626 --local-dir nari-labs/Dia-1.6B-0626
RUN conda run -n streamwise hf download nari-labs/Dia-1.6B --local-dir nari-labs/Dia-1.6B

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.sh .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate streamwise" >> ~/.bashrc

ENTRYPOINT ["bash", "run_httpserver.bash", "--dia"]
EXPOSE 8080