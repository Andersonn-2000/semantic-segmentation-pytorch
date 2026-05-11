FROM nvcr.io/nvidia/pytorch:22.11-py3

RUN rm -rf /workspace/*
WORKDIR /workspace/unet

COPY pyproject.toml .
COPY . .

RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir .