# .devcontainer/Dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Miniconda 설치
RUN apt update && apt install -y wget git nano curl bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init

ENV PATH=/opt/conda/bin:$PATH