FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN /bin/bash -c "apt update && \
    apt install -y \
    wget \
    bzip2 \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3.8 \
    python3-pip"

RUN git clone https://github.com/shuhei1110/dotfiles.git

RUN pip3 install --no-cache-dir -U pip
RUN pip3 install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html