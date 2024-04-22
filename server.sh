#!/usr/bin/env bash

docker run -it --rm --gpus=all --net=host -v ./tmp/triton_repo:/mnt --name triton-server nvcr.io/nvidia/tritonserver:23.09-py3
