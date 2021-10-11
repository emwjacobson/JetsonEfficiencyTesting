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
# R32 (release), REVISION: 6.1, GCID: 27863751, BOARD: t186ref, EABI: aarch64, DATE: Mon Jul 26 19:36:31 UTC 2021
```

```
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Feb_28_22:34:44_PST_2021
Cuda compilation tools, release 10.2, V10.2.300
Build cuda_10.2_r440.TC440_70.29663091_0
```

## Nano Info

### System Info

```
$ cat /etc/nv_tegra_release
# R32 (release), REVISION: 5.1, GCID: 26202423, BOARD: t210ref, EABI: aarch64, DATE: Fri Feb 19 16:45:52 UTC 2021
```

```
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_21:14:42_PDT_2019
Cuda compilation tools, release 10.2, V10.2.89
```

# Benchmarking Procedures

The `benchmark.cu` file is used for benchmarking the Jetson boards using various options.

Before each test, the CPU min/max frequency is set to it's maximum frequency (can also be changed later for more power usage info).

## AGX

Setting the CPU frequency:
```
AGX$ echo "2265600" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_{min,max}_freq
```

Setting the GPU frequency:

```
# All available frequencies: 114750000 216750000 318750000 420750000 522750000 624750000 675750000 828750000 905250000 1032750000 1198500000 1236750000 1338750000 1377000000
AGX$ echo "1377000000" | sudo tee /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/{min,max}_freq
```

## Nano

Set the CPU frequency:
```
Nano$ echo "1479000" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_{min,max}_freq
```

Set the GPU frequency:

```
# All available frequencies: 76800000 153600000 230400000 307200000 384000000 460800000 537600000 614400000 691200000 768000000 844800000 921600000
Nano$ echo "921600000" | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/{min,max}_freq
```

**Note** The fan ramp speed needs to be changed to make the fan more responsive when set.

```
$ echo "5" | sudo tee /sys/devices/pwm-fan/step_time
```

After the GPU and CPU frequencies have been set, the benchmark can be run.

```
$ sudo ./gpu_benchmark
```


# TODO
- ~~Square graphs, powers of 2 only~~
- ~~FLOPS 2nd graphs~~
- ~~Inference 1st and 3rd graphs~~
- ~~Export graphs~~
- Talk about architecture of Nano and AGX
	- GPU, DLAs
	- DFVS
	- FLOPS benchmark
	- Inference Server