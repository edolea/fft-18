FROM ubuntu:22.04

# Evita prompt durante installazioni
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

RUN apt-get update && apt-get install -y \
    cmake \
    git \
    curl \
    build-essential \
    libopencv-dev \
    libopenmpi-dev \
    openmpi-bin \
    libomp-dev \
    python3 \
    python3-pip \
    python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Preinstallazione pacchetti Python
COPY Python /workspace/Python
RUN python3 -m venv /workspace/.venv && \
    /workspace/.venv/bin/pip install --upgrade pip && \
    /workspace/.venv/bin/pip install numpy matplotlib

CMD ["/bin/bash"]