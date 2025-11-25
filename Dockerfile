# Dockerfile for Horizon X5 Model Conversion Toolchain
FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgfortran5 \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install essential Python packages
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install common ML/DL packages
RUN pip3 install --no-cache-dir \
    numpy==1.23.5 \
    opencv-python-headless==4.8.0.74 \
    pillow==9.5.0 \
    pyyaml==6.0 \
    protobuf==3.20.3 \
    onnx==1.14.0 \
    tensorflow==2.12.0

# Create workspace directory
WORKDIR /workspace

# Note: Horizon toolchain needs to be installed separately
# Instructions:
# 1. Download the Horizon X5 toolchain from: https://developer.horizon.ai/
# 2. Place the toolchain file (e.g., horizon_nn-*.whl) in the same directory as this Dockerfile
# 3. Uncomment the following lines and build the image:
# COPY horizon_nn-*.whl /tmp/
# RUN pip3 install /tmp/horizon_nn-*.whl && rm /tmp/horizon_nn-*.whl

# Set environment variables for the toolchain
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV PATH=/workspace:$PATH

# Create directories for models and output
RUN mkdir -p /workspace/models /workspace/output /workspace/calibration

# Set working directory
WORKDIR /workspace

CMD ["/bin/bash"]

