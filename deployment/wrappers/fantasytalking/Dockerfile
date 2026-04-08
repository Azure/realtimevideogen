# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v0.5.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
# Set to 0 to skip flash-attn installation (e.g. CPU-only CI builds)
ARG INSTALL_FLASH_ATTN=1
# Set to 1 to skip HuggingFace model downloads (e.g. CPU-only CI builds)
ARG SKIP_MODEL_DOWNLOAD=0
ENV DOCKER_REPO=${DOCKER_REPO}

# Install flash attention separately
RUN pip install psutil numpy
RUN if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
      pip install --no-build-isolation flash-attn==2.8.2 && \
      pip cache purge; \
    fi

# We use some of the Wan2.1 parallelism code
RUN git clone https://github.com/Wan-Video/Wan2.1.git wan && \
    cd wan && \
    pip install -r requirements.txt && \
    pip cache purge

RUN git clone https://github.com/Fantasy-AMAP/fantasy-talking.git fantasytalking
# cd fantasytalking
# pip install -r requirements.txt && \
# pip cache purge

WORKDIR /fantasytalking

# Use this instead of https://github.com/Fantasy-AMAP/fantasy-talking.git/requirements.txt
COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

# Download Fantasy Talking models from Hugging Face
RUN if [ "${SKIP_MODEL_DOWNLOAD}" != "1" ]; then \
      hf download Wan-AI/Wan2.1-I2V-14B-720P --local-dir /fantasytalking/Wan2.1-I2V-14B-720P && \
      hf download facebook/wav2vec2-base-960h --local-dir /fantasytalking/wav2vec2-base-960h && \
      hf download acvlab/FantasyTalking fantasytalking_model.ckpt --local-dir /fantasytalking; \
    else \
      echo "Skipping model download (CI build)"; \
    fi

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.sh .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


ENTRYPOINT ["bash", "run_httpserver.bash", "--fantasytalking"]
EXPOSE 8080
