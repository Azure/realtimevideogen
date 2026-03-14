# base image has CUDA, Miniconda, rtgen conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

# We avoid installing requirements.txt from the cloned repo to avoid diffusers overwrite
WORKDIR /hunyuanframepackvae

COPY docker_files/requirements.txt .
RUN conda run -n rtgen pip install -r requirements.txt && \
    conda run -n rtgen pip cache purge

# Download HunyuanVideo VAE model from Hugging Face
#RUN conda run -n rtgen hf download hunyuanvideo-community/HunyuanVideo --local-dir hunyuanvideo-community/HunyuanVideo \
#    --local-dir-use-symlinks False \
#    vae/config.json \
#    vae/diffusion_pytorch_model.safetensors
RUN conda run -n rtgen hf download hunyuanvideo-community/HunyuanVideo --local-dir hunyuanvideo-community/HunyuanVideo \
    vae/config.json \
    vae/diffusion_pytorch_model.safetensors

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate rtgen" >> ~/.bashrc

ENTRYPOINT ["bash", "run_httpserver.bash", "--hunyuanframepackvae"]
EXPOSE 8080
