# base image has CUDA, Miniconda, streamwise conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

# Install flash attention separately
RUN conda run -n streamwise pip install psutil numpy
RUN conda run -n streamwise pip install --no-build-isolation flash-attn==2.8.2 && \
    conda run -n streamwise pip cache purge

# We use some of the Wan2.1 parallelism code
RUN git clone https://github.com/Wan-Video/Wan2.1.git wan && \
    cd wan && \
    conda run -n streamwise pip install -r requirements.txt && \
    conda run -n streamwise pip cache purge

RUN git clone https://github.com/Fantasy-AMAP/fantasy-talking.git fantasytalking
# cd fantasytalking
# conda run -n streamwise pip install -r requirements.txt && \
# conda run -n streamwise pip cache purge

WORKDIR /fantasytalking

# Use this instead of https://github.com/Fantasy-AMAP/fantasy-talking.git/requirements.txt
COPY docker_files/requirements.txt .
RUN conda run -n streamwise pip install -r requirements.txt && \
    conda run -n streamwise pip cache purge

# Download Fantasy Talking models from Hugging Face
RUN conda run -n streamwise hf download Wan-AI/Wan2.1-I2V-14B-720P --local-dir /fantasytalking/Wan2.1-I2V-14B-720P && \
    conda run -n streamwise hf download facebook/wav2vec2-base-960h --local-dir /fantasytalking/wav2vec2-base-960h && \
    conda run -n streamwise hf download acvlab/FantasyTalking fantasytalking_model.ckpt --local-dir /fantasytalking

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.sh .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate streamwise" >> ~/.bashrc

ENTRYPOINT ["bash", "run_httpserver.bash", "--fantasytalking"]
EXPOSE 8080
