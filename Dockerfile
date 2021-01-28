ARG UBUNTU_VERSION=18.04
ARG CUDA=10.1

FROM nvidia/cuda:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base

MAINTAINER <25718561+dmitry-vorobiev@users.noreply.github.com>

WORKDIR /io
WORKDIR /app

ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3 \
    python3-pip

# Install latest Git LFS:
# https://github.com/git-lfs/git-lfs/wiki/Installation
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get update && apt-get install -y git-lfs

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    wheel

RUN python3 -m pip install --no-cache-dir \
    torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . /app/

RUN python3 -m pip install -r /app/requirements.txt

RUN cd /io && \
    git clone https://github.com/RossiyaSegodnya/ria_news_dataset && \
    cd ria_news_dataset && \
    git lfs install && \
    git lfs pull

RUN cd /io && \
    git clone https://huggingface.co/dmitry-vorobiev/rubert_ria_headlines && \
    cd rubert_ria_headlines && \
    git lfs install && \
    git lfs pull
