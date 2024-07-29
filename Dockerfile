# Use the official Python 3.11 image as a base image
FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

SHELL ["/bin/bash", "-c"]
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    --mount=type=cache,target=/root/.cache \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt update && \
    apt install -y --no-install-recommends build-essential &&\
    pip install --upgrade pip twine wheel setuptools &&\
    pip install -e .[dev] && \
    rm -rf /var/lib/apt/lists/*
