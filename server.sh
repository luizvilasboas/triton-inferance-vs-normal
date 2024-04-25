#!/usr/bin/env bash

docker run -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 --gpus all -v ./tmp/triton_repo:/mnt -v ./mds:/models --name triton-server nvcr.io/nvidia/tritonserver:23.09-py3
