# Inference Testing

- [Inference Testing](#inference-testing)
  - [Jetson Setup](#jetson-setup)
  - [Downloading the Triton Inference Server](#downloading-the-triton-inference-server)
  - [Perf Analyzer](#perf-analyzer)
  - [In Progress Work](#in-progress-work)

## Jetson Setup

As of Triton release [2.12.0](https://github.com/triton-inference-server/server/releases/tag/v2.12.0), the Triton Inference Server can be run on Jetson devices.


## Downloading the Triton Inference Server

**Be sure to install the dependencies listed in the [Triton Release Notes](https://github.com/triton-inference-server/server/releases/tag/v2.12.0) before running!**

The inference server and models can be easily downloaded by running the `setup.sh` script.

Finally the inference server can be ran by running:

`$ ./tritonserver/bin/tritonserver --model-repository=./models/ --backend-directory=./tritonserver/backends/ --backend-config=tensorflow,version=2 --model-control-mode=explicit --load-model=mobilenet`

## Perf Analyzer

The perf analyzer can be used to measure inference performance.

If you are **not** running the triton server locally, this command can be used to benchmark a single model. It uses the C API for maximum performance.
`$ ./tritonserver/clients/bin/perf_analyzer --triton-server-directory=./tritonserver --model-repository=./models/ --service-kind=triton_c_api -m mobilenet`

If you **are** running the triton server in another instance, this command can be used to benchmark it. It uses the HTTP protocol for a potentially more realistic test.
`$ ./tritonserver/clients/bin/perf_analyzer --shared-memory=system -m mobilenet`



## In Progress Work

https://github.com/triton-inference-server/client
https://github.com/triton-inference-server/client/tree/main/src/python/examples

https://github.com/triton-inference-server/server/blob/main/docs/perf_analyzer.md

https://github.com/onnx/models

`/usr/src/tensorrt/bin/trtexec --onnx=models/mobilenet/1/model.onnx --saveEngine=mobilenet_engine.trt --device=0`
`/usr/src/tensorrt/bin/trtexec --onnx=models/mobilenet/1/model.onnx --saveEngine=mobilenet_engine.trt --useDLACore=0 --allowGPUFallback`

TODO:
- AGX
  - Fixed batch, varying frequency
  - Batch Sizes: 1, 2, 4, 8
- Nano
  - Fixed batch, varying frequency
  - Batch Sizes: 1, 2, 4, 8
