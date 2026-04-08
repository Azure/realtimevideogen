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

RUN git clone https://github.com/lllyasviel/FramePack.git hunyuanframepackf1

# We avoid installing requirements.txt from the cloned repo to avoid diffusers overwrite
WORKDIR /hunyuanframepackf1

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

# Download HunyuanVideo and FramePackI2V_HY models from Hugging Face
RUN if [ "${SKIP_MODEL_DOWNLOAD}" != "1" ]; then \
      hf download hunyuanvideo-community/HunyuanVideo --local-dir hunyuanvideo-community/HunyuanVideo && \
      hf download lllyasviel/FramePack_F1_I2V_HY_20250503 --local-dir lllyasviel/FramePack_F1_I2V_HY_20250503; \
    else \
      echo "Skipping model download (CI build)"; \
    fi

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


ENTRYPOINT ["bash", "run_httpserver.bash", "--hunyuanframepackf1"]
EXPOSE 8080
