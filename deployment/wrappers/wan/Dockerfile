# base image has CUDA, Miniconda, rtgen conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

# Install flash attention separately
RUN conda run -n rtgen pip install psutil numpy
RUN conda run -n rtgen pip install --no-build-isolation flash-attn==2.8.2 && \
    conda run -n rtgen pip cache purge

RUN git clone https://github.com/Wan-Video/Wan2.1.git wan && \
    cd wan && \
    conda run -n rtgen pip install -r requirements.txt && \
    conda run -n rtgen pip cache purge

WORKDIR /wan

RUN conda run -n rtgen hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /wan/Wan2.1-I2V-14B-480P && \
    conda run -n rtgen hf download Wan-AI/Wan2.1-I2V-14B-720P --local-dir /wan/Wan2.1-I2V-14B-720P

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

COPY docker_files/vae.py /wan/wan/modules/vae.py

SHELL ["/bin/bash", "-c"]
RUN echo "source activate rtgen" >> ~/.bashrc

# docker run --gpus all -p 18080:8080 --name wan_rest wan
# docker exec -it wan_rest ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--wan", "--wan_variation", "480p"]
EXPOSE 8080
