#!/usr/bin/env bash

YOLO_VERSION="yolov8n.pt"
YOLO_REPO_NAME="yolov8n"

yolo_export_command="yolo export model=${YOLO_VERSION} format=onnx dynamic=False device=0"
eval "$yolo_export_command"

mkdir -p tmp/triton_repo/${YOLO_REPO_NAME}/1 || {
  echo "Error creating directory: tmp/triton_repo/${YOLO_REPO_NAME}/1"
  exit 1
}

cp "${YOLO_REPO_NAME}.onnx" tmp/triton_repo/${YOLO_REPO_NAME}/1/model.onnx || {
  echo "Error copying YOLO model: ${YOLO_REPO_NAME}.onnx"
  exit 1
}

cp config.pbtxt.sample tmp/triton_repo/${YOLO_REPO_NAME}/config.pbtxt || {
  echo "Error copying configuration file: config.pbtxt.sample"
  exit 1
}
