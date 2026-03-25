# base image has CUDA, Miniconda, streamwise conda env with torch, and base_requirements
ARG DOCKER_REPO=reposmustbeset.azurecr.io
ARG BASE_TAG=v1.0.0

FROM ${DOCKER_REPO}/base:${BASE_TAG}

ARG DOCKER_REPO
ENV DOCKER_REPO=${DOCKER_REPO}

# Clone 4KAgent
RUN git clone --depth 1 https://github.com/taco-group/4KAgent /4kagent/4KAgent

WORKDIR /4kagent

# Create 4kagent conda environment (Python 3.10, older torch required by 4KAgent)
RUN conda create -n 4kagent python=3.10 -y && \
    conda run -n 4kagent pip install \
        torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    conda run -n 4kagent pip install -r /4kagent/4KAgent/installation/requirements.txt && \
    conda run -n 4kagent pip install --upgrade transformers==4.49.0 && \
    conda run -n 4kagent pip cache purge

# Create tool conda environments for 4KAgent executors
RUN conda create -n 4kagent_spec_tools python=3.8 -y && \
    conda run -n 4kagent_spec_tools pip install \
        -r /4kagent/4KAgent/installation/spec_tools_requirements.txt && \
    conda run -n 4kagent_spec_tools pip cache purge

RUN conda create -n 4kagent_dehaze python=3.8 -y && \
    conda run -n 4kagent_dehaze pip install \
        torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
        --extra-index-url https://download.pytorch.org/whl/cu117 && \
    conda run -n 4kagent_dehaze pip install \
        -r /4kagent/4KAgent/executor/dehazing/tools/RIDCP_dehazing/requirements.txt && \
    conda run -n 4kagent_dehaze pip cache purge

RUN conda create -n 4kagent_sr python=3.10 -y && \
    conda run -n 4kagent_sr pip install \
        -r /4kagent/4KAgent/executor/super_resolution/tools/OSEDiff/requirements.txt && \
    conda run -n 4kagent_sr pip cache purge

# Create tool symbolic links
RUN cd /4kagent/4KAgent && bash installation/tools_sym_links.sh

# Download pretrained checkpoints from Hugging Face
RUN conda run -n 4kagent huggingface-cli download YSZuo/4KAgent-Toolbox-Pretrained-Models \
        --local-dir /4kagent/4KAgent/pretrained_ckpts && \
    cp /4kagent/4KAgent/pretrained_ckpts/BOBL/Face_Detection/shape_predictor_68_face_landmarks.dat \
       /4kagent/4KAgent/executor/old_photo_restoration/tools/BOBL/Face_Detection/ && \
    cp /4kagent/4KAgent/pretrained_ckpts/hpsv2/bpe_simple_vocab_16e6.txt.gz \
       /4kagent/4KAgent/utils/clib_fiqa/model/ && \
    conda run -n 4kagent bash -c \
        'cp /4kagent/4KAgent/pretrained_ckpts/hpsv2/bpe_simple_vocab_16e6.txt.gz \
         "$(python -c "import hpsv2, os; print(os.path.join(os.path.dirname(hpsv2.__file__), '"'"'src/open_clip'"'"'))")"'

COPY docker_files/requirements.txt .
RUN conda run -n streamwise pip install -r requirements.txt && \
    conda run -n streamwise pip cache purge

COPY docker_files/*.py .
COPY docker_files/*.bash .
COPY docker_files/*.json .
COPY docker_files/templates ./templates
COPY docker_files/static ./static

SHELL ["/bin/bash", "-c"]
RUN echo "source activate streamwise" >> ~/.bashrc

ENTRYPOINT ["bash", "run_httpserver.bash", "--4kagent"]
EXPOSE 8080
