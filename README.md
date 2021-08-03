# Local Environment Setup

Create a virtual environment for Python to run in:

`$ python3.8 -m venv .venv`

Activate the virtual environment

`$ source .venv/bin/activate`

Update pip and setuptools

`$ pip install --upgrade pip setuptools`

Install requirements

`$ pip install -r requirements.txt`


# Research

The initial goal is to determine the different variables that we can change to see how effieiency changes.
As of now, these are:
- GPU Frequency
- CPU Frequency
- Memory Frequency
- Matrix Size
- Deep Learning Accelerators (DLAs)
- Tensor Cores
- Data Types


## AGX Info

### System Info

```
$ cat /etc/nv_tegra_release
# R32 (release), REVISION: 4.4, GCID: 23942405, BOARD: t186ref, EABI: aarch64, DATE: Fri Oct 16 19:37:08 UTC 2020
```

```
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_21:14:42_PDT_2019
Cuda compilation tools, release 10.2, V10.2.89
```

## Nano Info

### System Info

```
todo
```

# Benchmarking Procedures

The `benchmark.cu` file is used for benchmarking the Jetson boards using various options.

Before each test, the CPU min/max frequency is set to it's maximum frequency (can also be changed later for more power usage info).

```
AGX$ echo "2265600" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_{min,max}_freq
```

The GPU frequency is then set

### AGX
```
# All available frequencies: 114750000 216750000 318750000 420750000 522750000 624750000 675750000 828750000 905250000 1032750000 1198500000 1236750000 1338750000 1377000000
AGX$ echo "1377000000" | sudo tee /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/{min,max}_freq
```

**Note** On the AGX, the fan ramp speed needs to be changed to make the fan more responsive when set.

```
AGX$ echo "5" | sudo tee /sys/devices/pwm-fan/step_time
```

### Nano
```
Nano$ todo
```

After the GPU and CPU frequencies have been set, the benchmark can be run.

```
$ sudo ./gpu_benchmark
```


# TODO
- Start work on Nano
  - Flops/Watt for all frequencies (square matricies)
  - Flops/Watt for most efficient & highest (non-square)
- Triton Inference on AGX/Nano