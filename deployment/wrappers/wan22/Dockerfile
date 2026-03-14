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

RUN git clone https://github.com/Wan-Video/Wan2.2.git wan22 && \
    cd wan22 && \
    conda run -n streamwise pip install -r requirements.txt

WORKDIR /wan22

RUN conda run -n streamwise hf download Wan-AI/Wan2.2-I2V-A14B --local-dir /wan/Wan2.2-I2V-A14B

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate streamwise" >> ~/.bashrc

# docker run --gpus all -p 18080:8080 --name wan22_rest wan
# docker exec -it wan22_rest ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--wan22"]
EXPOSE 8080
