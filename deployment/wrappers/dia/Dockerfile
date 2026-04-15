# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v0.5.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

RUN git clone https://github.com/nari-labs/dia.git

WORKDIR /dia

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

RUN hf download nari-labs/Dia-1.6B-0626 --local-dir nari-labs/Dia-1.6B-0626
RUN hf download nari-labs/Dia-1.6B --local-dir nari-labs/Dia-1.6B

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/*.sh .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


ENTRYPOINT ["bash", "run_httpserver.bash", "--dia"]
EXPOSE 8080