# base image has CUDA, Miniconda, streamwise conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
# Set to 0 to skip flash-attn installation (e.g. CPU-only CI builds)
ARG INSTALL_FLASH_ATTN=1
ENV DOCKER_REPO=${DOCKER_REPO}

RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake mediainfo && rm -rf /var/lib/apt/lists/*

# Install flash attention separately (requires CUDA; skip with INSTALL_FLASH_ATTN=0)
RUN conda run -n streamwise pip install psutil numpy
RUN if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
      conda run -n streamwise pip install --no-build-isolation flash-attn==2.8.2 && \
      conda run -n streamwise pip cache purge; \
    fi

WORKDIR /kokoro

COPY docker_files/requirements.txt .
RUN conda run -n streamwise pip install -r requirements.txt && \
    conda run -n streamwise pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate streamwise" >> ~/.bashrc

# docker run --gpus all -p 18080:8080 kokoro
# docker exec -it <container-id> ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--kokoro"]
EXPOSE 8080