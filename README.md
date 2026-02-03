# Add kernel experiments

Some experiments on writing a basic CUDA kernel with C++, consisting in adding two arrays of floats of length $n$,
following $y: = x + y$. My goal was to benchmark the performance of different optimization techniques in kernel writing:
signed index, `__restrict__` pointers, vectorisation, rolling with and without tail, ... (see 
[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) inter alia).

## Benchmarking

### Effective Bandwidth

The kernels are compared based on their effective bandwidth in $\text{GB/s}$. Each kernel is performing $2n$ read operations 
($x$ and $y$) and $n$ write operation ($y$), so the effective bandwidth is $2n \times s/t$ where
- $t$ is the time spent in the kernel ;
- $s$ is the format size: $s = 4\; \text B$ for `float` (FP32) and $s = 2\; \text B$ for `__nv_bfloat16` (BF16)

### Hardware & Theoretical bandwidth

Some experiments were run on my laptop, with a _NVIDIA Geforce RTX 4060 Laptop_ GPU. According to
[NotebookCheck.net](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4060-Laptop-GPU-Benchmarks-and-Specs.675692.0.html),
it has a clock speed of 16 Gbps (effective) and a 128 Bit memory bus, so the effective bandwidth is 
$$16 \times 128 / 8 = 256\; \text{GB/s}$$


## Results

I run the experiments with $n = 2^{27} = 13 4217 728$ and $\texttt{blockSize} = 256$.

#### Laptop (RTX 4060 laptop), plugged in (AC power)

```text
addNaive (FP32)                             2987.939 ms  |       0.13 GB/s  |      0.000e+00
addNaiveRestrict (FP32)                      925.093 ms  |       0.44 GB/s  |      0.000e+00
addNaiveSizeT (FP32)                        1678.231 ms  |       0.24 GB/s  |      0.000e+00
addNaiveSizeTRestrict (FP32)                1378.039 ms  |       0.29 GB/s  |      0.000e+00
addNaiveFloat2Restrict (FP32)               1121.469 ms  |       0.36 GB/s  |      0.000e+00
addNaiveFloat4Restrict (FP32)                950.717 ms  |       0.42 GB/s  |      0.000e+00
addNaiveFloat4NoTail (FP32)                  911.890 ms  |       0.44 GB/s  |      0.000e+00
addNaiveFloat2RestrictNoTail (FP32)         1132.483 ms  |       0.36 GB/s  |      0.000e+00
addNaiveFloat4RestrictNoTail (FP32)          973.865 ms  |       0.41 GB/s  |      0.000e+00

addThreadBlock (FP32)                          7.171 ms  |     224.60 GB/s  |      0.000e+00
addThreadBlockRestrict (FP32)                  7.175 ms  |     224.46 GB/s  |      0.000e+00
```

more TODO (real kernels).


#### Laptop (RTX 4060 laptop), on battery

TODO :)