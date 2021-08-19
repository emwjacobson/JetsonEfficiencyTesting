#!/bin/bash

download_alexnet() {
  wget 'https://github.com/onnx/models/raw/master/vision/classification/alexnet/model/bvlcalexnet-9.onnx' -O models/alexnet/1/model.onnx;
}

download_mobilenet() {
  wget 'https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx' -O models/mobilenet/1/model.onnx;
}

download_all() {
  download_alexnet;
  download_mobilenet;
}

download_all;