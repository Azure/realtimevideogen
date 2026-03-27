# base image has CUDA and Python with torch and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

RUN git clone https://github.com/FoundationVision/LlamaGen llamagen

WORKDIR /llamagen

# Download LlamaGen model from Hugging Face
RUN huggingface-cli download FoundationVision/LlamaGen --local-dir FoundationVision/LlamaGen
#RUN huggingface-cli download DeepFloyd/t5-v1_1-xxl --local-dir DeepFloyd/t5-v1_1-xxl
RUN huggingface-cli download google/flan-t5-xl --local-dir google/flan-t5-xl \
    config.json \
    special_tokens_map.json \
    spiece.model \
    tokenizer_config.json \
    pytorch_model.bin.index.json \
    pytorch_model-00001-of-00002.bin \
    pytorch_model-00002-of-00002.bin
RUN huggingface-cli download peizesun/llamagen_t2i --local-dir peizesun/llamagen_t2i

COPY docker_files/requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static


# docker run --gpus all -p 18080:8080 --name llamagen_rest llamagen
# docker exec -it llamagen_rest ls -l /
ENTRYPOINT ["bash", "run_httpserver.bash", "--llamagen"]
EXPOSE 8080