#!/usr/bin/env bash

YOLO_VERSION="yolov8n.pt"
YOLO_REPO_NAME="yolo8n"

yolo_export_command="yolo export model=${YOLO_VERSION} format=onnx dynamic=True device=0 batch=4"
eval "$yolo_export_command"

mkdir -p tmp/triton_repo/1 || {
  echo "Error creating directory: tmp/triton_repo/${YOLO_REPO_NAME}/1"
  exit 1
}

cp "${YOLO_REPO_NAME}.onnx" tmp/triton_repo/1 || {
  echo "Error copying YOLO model: ${YOLO_REPO_NAME}.onnx"
  exit 1
}

cp config.pbtxt.sample tmp/triton_repo/ || {
  echo "Error copying configuration file: config.pbtxt.sample"
  exit 1
}
