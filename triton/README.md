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

IN PROGRESS
