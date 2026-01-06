# RunPod serverless worker image
# Note: choose a CUDA base compatible with the GPU types you deploy on RunPod.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY inference.py handler.py ./

# RunPod expects your container to start the worker (runpod serverless)
CMD ["python3", "-u", "handler.py"]
