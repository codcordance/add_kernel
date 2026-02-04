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
addNaiveIV1U1NC (FP32)                     11562.777 ms  |       0.14 GB/s  |      0.000e+00
addNaiveIV1U1NR (FP32)                      3227.047 ms  |       0.50 GB/s  |      0.000e+00
addNaiveSV1U1NC (FP32)                      6470.865 ms  |       0.25 GB/s  |      0.000e+00
addNaiveSV1U1NR (FP32)                      5181.450 ms  |       0.31 GB/s  |      0.000e+00
addNaiveIV1U2TC (FP32)                      6468.836 ms  |       0.25 GB/s  |      0.000e+00
addNaiveIV1U2TR (FP32)                      4678.101 ms  |       0.34 GB/s  |      0.000e+00
addNaiveIV1U2MC (FP32)                      6469.069 ms  |       0.25 GB/s  |      0.000e+00
addNaiveIV1U2MR (FP32)                      4678.330 ms  |       0.34 GB/s  |      0.000e+00
addNaiveSV1U2TC (FP32)                      6459.227 ms  |       0.25 GB/s  |      0.000e+00
addNaiveSV1U2TR (FP32)                      4712.350 ms  |       0.34 GB/s  |      0.000e+00
addNaiveSV1U2MC (FP32)                      6467.008 ms  |       0.25 GB/s  |      0.000e+00
addNaiveSV1U2MR (FP32)                      4819.595 ms  |       0.33 GB/s  |      0.000e+00
addNaiveIV1U4TC (FP32)                     11573.858 ms  |       0.14 GB/s  |      0.000e+00
addNaiveIV1U4TR (FP32)                      3608.495 ms  |       0.45 GB/s  |      0.000e+00
addNaiveIV1U4MC (FP32)                     11571.075 ms  |       0.14 GB/s  |      0.000e+00
addNaiveIV1U4MR (FP32)                      3222.539 ms  |       0.50 GB/s  |      0.000e+00
addNaiveSV1U4TC (FP32)                     11254.399 ms  |       0.14 GB/s  |      0.000e+00
addNaiveSV1U4TR (FP32)                      3283.462 ms  |       0.49 GB/s  |      0.000e+00
addNaiveSV1U4MC (FP32)                     11295.900 ms  |       0.14 GB/s  |      0.000e+00
addNaiveSV1U4MR (FP32)                      3249.226 ms  |       0.50 GB/s  |      0.000e+00
addNaiveIV2U1TC (FP32)                      8449.339 ms  |       0.19 GB/s  |      0.000e+00
addNaiveIV2U1TR (FP32)                      3575.265 ms  |       0.45 GB/s  |      0.000e+00

addThreadBlock (FP32)                          7.171 ms  |     224.59 GB/s  |      0.000e+00
addThreadBlockRestrict (FP32)                  7.165 ms  |     224.78 GB/s  |      0.000e+00
```

more TODO :)


#### Laptop (RTX 4060 laptop), on battery

TODO :)