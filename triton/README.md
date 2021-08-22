# Inference Testing

- [Inference Testing](#inference-testing)
  - [Jetson Setup](#jetson-setup)
  - [Downloading the Triton Inference Server](#downloading-the-triton-inference-server)
  - [Perf Analyzer](#perf-analyzer)

## Jetson Setup

As of Triton release [2.12.0](https://github.com/triton-inference-server/server/releases/tag/v2.12.0), the Triton Inference Server can be run on Jetson devices.

The `tritonserver2.12.0-jetpack4.6.tgz` file must be downloaded from the [release notes](https://github.com/triton-inference-server/server/releases/tag/v2.12.0) and extracted on the Jetson device.


## Downloading the Triton Inference Server

**Be sure to also install the dependencies listed in the notes before running!**

The inference server and models can be easily downloaded by running the `setup.sh` script.

Finally the inference server can be ran by running:

`$ ./tritonserver/bin/tritonserver --model-repository=./models/ --backend-directory=./tritonserver/backends/ --backend-config=tensorflow,version=2`

## Perf Analyzer

The perf analyzer can be used to measure inference performance.

`$ ./tritonserver/clients/bin/perf_analyzer -m mobilenet --triton-server-directory=./tritonserver --model-repository=./models/ --service-kind=triton_c_api`

https://github.com/triton-inference-server/client
https://github.com/triton-inference-server/client/tree/main/src/python/examples

https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md

https://github.com/onnx/models

TODO:
- Figure out what model(s) to use for testing
  - AlexNet
    - Almost fills ram, but works (might need to reboot beforehand) (3.8/4GB RAM)
    - ~ 1.2GB GPU RAM
  - MobileNet
    - Pretty lightweight
    - ~775MB GPU RAM
  - GoogLeNet
    - Gets to 3.5/4GB full system RAM usage
    - ~1.5GB GPU RAM
  - TinyYolov2
    - Gets to 3.4/4GB full system RAM usage
    - ~1.2GB GPU RAM
  - Yolov4
    - Pretty large model, has to use swap (3.7GB system + 1.5GB swap)
    - ~2.2GB GPU RAM
  - MNIST
    - Small model, 2.6/4GB full system RAM usage
    - ~640MB GPU RAM
- Determine concurrency range
- Convert model to TensorRT engine
  - Test on GPU and DLAs
- Shared Memory settings in perf_analyzer: system/cuda/none
  - Not enabled in the C engine :/

- Inference Server
- Power usage