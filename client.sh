#!/usr/bin/env bash

docker run -it --rm --gpus=all --net=host -v ./tmp/triton_repo:/mnt --name triton-client nvcr.io/nvidia/tritonserver:21.10-py3-sdk
