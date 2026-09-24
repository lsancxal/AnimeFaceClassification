# CUDA PyTorch base for Linux GPU training
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .

# Base image already has torch/torchvision; install the remaining packages from requirements.txt
RUN grep -vE '^\s*(#|$|torch>=|torchvision>=|torch==|torchvision==)' requirements.txt \
        > /tmp/requirements.docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements.docker.txt \
    && rm /tmp/requirements.docker.txt

COPY . .

RUN mkdir -p /app/outputs /app/src/data/extracted

CMD ["python", "main.py"]
