#https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html
FROM nvcr.io/nvidia/pytorch:24.08-py3 AS lightstream

# === Configure environment variables ===
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt update && \
    apt install -y  \
    openssh-server \ 
    rsync \
    wget \
    bzip2 zip unzip \
    libvips libvips-tools libvips-dev 

RUN pip3 install --upgrade pip && \
    pip3 install lightning && \
    pip3 install dataclasses_json && \
    pip3 install lightstream && \
    pip3 install albumentationsxl && \
    pip3 install pyvips

# === Lightstream image ===
FROM lightstream AS streamingclam
# Copy the source code
WORKDIR /app
COPY . .
