# base image has CUDA, Miniconda, rtgen conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

WORKDIR /realesrgan

# RUN conda run -n rtgen pip install git+https://github.com/ai-forever/Real-ESRGAN.git
RUN conda run -n rtgen pip install setuptools wheel && \
    conda run -n rtgen pip install --no-build-isolation git+https://github.com/ai-forever/Real-ESRGAN.git

COPY docker_files/requirements.txt .
RUN conda run -n rtgen pip install -r requirements.txt && \
    conda run -n rtgen pip cache purge

# Download Real-ESRGAN model from Hugging Face
RUN conda run -n rtgen huggingface-cli download ai-forever/Real-ESRGAN --local-dir ai-forever/Real-ESRGAN

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate rtgen" >> ~/.bashrc

ENTRYPOINT ["bash", "run_httpserver.bash", "--realesrgan"]
EXPOSE 8080
