#!/bin/bash

download_triton() {
  wget -nc 'https://github.com/triton-inference-server/server/releases/download/v2.12.0/tritonserver2.12.0-jetpack4.6.tgz' -O tritonserver.tgz;
  mkdir tritonserver;
  tar -xvf tritonserver.tgz -C tritonserver/;
  cp tritonserver/bin/tritonserver tritonserver/;
  rm tritonserver.tgz;
}

# Image Classification
download_alexnet() {
  mkdir -p models/alexnet/1/;
  wget -nc 'https://github.com/onnx/models/raw/master/vision/classification/alexnet/model/bvlcalexnet-9.onnx' -O models/alexnet/1/model.onnx;
}

download_mobilenet() {
  mkdir -p models/mobilenet/1/;
  wget -nc 'https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx' -O models/mobilenet/1/model.onnx;
}

download_googlenet() {
  mkdir -p models/googlenet/1/;
  wget -nc 'https://github.com/onnx/models/raw/master/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx' -O models/googlenet/1/model.onnx;
}


# Object Detection
download_tinyyolov2() {
  mkdir -p models/tinyyolov2/1/;
  wget -nc 'https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx' -O models/tinyyolov2/1/model.onnx;
}

download_yolov4() {
  mkdir -p models/yolov4/1/;
  wget -nc 'https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx' -O models/yolov4/1/model.onnx;
}


# MNIST
download_mnist() {
  mkdir -p models/mnist/1/;
  wget -nc 'https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-8.onnx' -O models/mnist/1/model.onnx;
}


download_all() {
  download_triton;

  download_alexnet;
  download_mobilenet;
  download_googlenet;

  download_tinyyolov2;
  download_yolov4;

  download_mnist;
}

download_all;