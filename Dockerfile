ARG UBUNTU_VERSION=18.04
ARG CUDA=10.1

FROM nvidia/cuda:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base

MAINTAINER <25718561+dmitry-vorobiev@users.noreply.github.com>

WORKDIR /io
WORKDIR /app

ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    wheel

RUN python3 -m pip install --no-cache-dir \
    torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . /app/

RUN python3 -m pip install -r /app/requirements.txt
