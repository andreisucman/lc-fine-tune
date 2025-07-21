# Use NVIDIA CUDA base image with Python 3.10 and CUDA 12.6
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Install python, pip, and other dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-venv python3-dev build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 and pip3 for convenience
RUN ln -s /usr/bin/python3 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

# Set CUDA_HOME environment variable
ENV CUDA_HOME=/usr/local/cuda

# Upgrade pip and install python packages
WORKDIR /app
COPY requirements.txt .
COPY main.py .

RUN pip install --upgrade pip setuptools wheel packaging torch --extra-index-url https://download.pytorch.org/whl/cu126 \
    && pip install flash-attn --no-build-isolation \
    && pip install -r requirements.txt

CMD ["python", "main.py"]
