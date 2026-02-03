# Add kernel experiments

Some experiments on writing a basic CUDA kernel with C++, consisting in adding two arrays of floats of length $n$,
following $y: = x + y$. My goal was to benchmark the performance of different optimization techniques in kernel writing:
signed index, `__restrict__` pointers, vectorisation, rolling with and without tail, ... (see 
[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) inter alia).

## Benchmarking

### Effective Bandwidth

The kernels are compared based on their effective bandwidth in $\text{GB/s}$. Each kernel is performing $2n$ read
operations and $n$ write operations, so the effective bandwidth is $2n \times s/t$ where
- $t$ is the time spent in the kernel ;
- $s$ is the format size: $s = 4\ \text B$ for `float` (FP32) and $s = 2\ \text B$ for `__nv_bfloat16` (BF16)

### Hardware & Theoretical bandwidth

Some experiments were run on my laptop, with a _NVIDIA Geforce RTX 4060 Laptop_ GPU. According to
[NotebookCheck.net](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4060-Laptop-GPU-Benchmarks-and-Specs.675692.0.html),
it has a clock speed of 16 Gbps (effective) and a 128 Bit memory bus, so the theoretical bandwidth is 
$$16 \times 128 / 8 = 256\ \text{GB/s}$$


## Results

I run the following experiment:
- $n = 2^{27} = 134\ 217\ 728$ (so an array of $n$ `float`/FP32 is $\approx 0.54\ \text{GB}$) ;
- $\texttt{blockSize} = 256$ ;
- each kernel is run $r = 4$ times.

#### Laptop (RTX 4060 laptop), plugged in (AC power)

table (kernel, mean duration, mean effective bandwidth, total error)
```text
addNaive (FP32)                            11551.580 ms  |       0.14 GB/s  |      0.000e+00
addNaiveRestrict (FP32)                     3225.731 ms  |       0.50 GB/s  |      0.000e+00
addNaiveSizeT (FP32)                        6471.052 ms  |       0.25 GB/s  |      0.000e+00
addNaiveSizeTRestrict (FP32)                5145.722 ms  |       0.31 GB/s  |      0.000e+00
addNaiveFloat2Restrict (FP32)               4106.704 ms  |       0.39 GB/s  |      0.000e+00
addNaiveFloat4Restrict (FP32)               3438.462 ms  |       0.47 GB/s  |      0.000e+00
addNaiveFloat4NoTail (FP32)                 3352.914 ms  |       0.48 GB/s  |      0.000e+00
addNaiveFloat2RestrictNoTail (FP32)         4105.456 ms  |       0.39 GB/s  |      0.000e+00
addNaiveFloat4RestrictNoTail (FP32)         3353.073 ms  |       0.48 GB/s  |      0.000e+00

addThreadBlock (FP32)                          7.171 ms  |     224.59 GB/s  |      0.000e+00
addThreadBlockRestrict (FP32)                  7.165 ms  |     224.78 GB/s  |      0.000e+00
```

more TODO :)


#### Laptop (RTX 4060 laptop), on battery

TODO :)