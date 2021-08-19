# Inference Testing

- [Inference Testing](#inference-testing)
  - [Jetson Setup](#jetson-setup)
  - [Client Setup](#client-setup)

## Jetson Setup

As of Triton release [2.12.0](https://github.com/triton-inference-server/server/releases/tag/v2.12.0), the Triton Inference Server can be run on Jetson devices.

The `tritonserver2.12.0-jetpack4.6.tgz` file must be downloaded from the [release notes](https://github.com/triton-inference-server/server/releases/tag/v2.12.0) and extracted on the Jetson device.

**Be sure to also install the dependencies listed in the notes before running!**

The triton server git repo can also be cloned in the same directory to help with the downloading of models.

`$ git clone https://github.com/triton-inference-server/server.git`

Then models downloaded by running:

```
$ cd server/docs/examples/
$ ./fetch_models.sh
```

Finally the inference server can be ran by running:

`$ ./bin/tritonserver --model-repository=./server/docs/examples/model_repository/ --backend-directory=./backends/ --backend-config=tensorflow,version=2`

This should create a GRPC server on port 8001, and an HTTP server on port 8000.

## Client Setup

Create a virtual environment for Python to run in:

`$ python3.8 -m venv .venv`

Activate the virtual environment

`$ source .venv/bin/activate`

Update pip and setuptools

`$ pip install --upgrade pip setuptools`

Install requirements

```
$ pip install nvidia-pyindex
$ pip install tritonclient[all]
$ pip install pillow attrdict
```

## Perf Analyzer

The perf analyzer can be used to measure inference performance.

`$ ./clients/bin/perf_analyzer -m densenet_onnx --service-kind=triton_c_api --triton-server-directory=. --model-repository=./server/docs/examples/model_repository --concurrency-range 4:16:4 -p 20000`

https://github.com/triton-inference-server/client
https://github.com/triton-inference-server/client/tree/main/src/python/examples

https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md

https://github.com/onnx/models

TODO:
- Figure out what model(s) to use for testing
  - AlexNet, GoogleNet, and ResNet-50 (Good for DLAs according to https://www.assured-systems.com/us/news/article/nvidia-jetson-agx-xavier-for-new-era-of-ai-in-robotics/)
  - Image, Text, Mnist
  - 5-7 nets
- Determine concurrency range
- Convert model to TensorRT engine
  - Test on GPU and DLAs

- Inference Server
- Power usage