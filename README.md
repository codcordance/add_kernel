# Add kernel experiments

Some experiments on writing a basic CUDA kernel with C++, consisting in adding two arrays of floats of length $n$,
following $y: = x + y$. My goal was to follow the approach presented in the [CUDA C++ tutorial](https://docs.nvidia.com/cuda/cuda-c-tutorial/index.html)
to [optimize](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#optimize) the kernel at best.

### Hardware & Theoretical bandwidth

Some experiments were run on my laptop, with an _RTX 4060 GPU (laptop version)_.


I run the experiments with $n = 2^{25} = 33\ 554\ 432$ and $\texttt{blockSize} = 256$.

#### Laptop (RTX 4060 laptop), plugged in (AC power)

```text
Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)   Category                                   Operation                               
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  -----------  -----------------------------------------------------------------------
     96.1      29219323847         10  2921932384.7  2922872725.5  2886322536  2964987954   22085809.4  CUDA_KERNEL  add_naive_kernel(int, const float *, float *)                          
      3.0        917981253         10    91798125.3    67687496.0    49484641   181808069   50006193.9  CUDA_KERNEL  add_block_kernel(int, const float *, float *)                          
      0.5        153213380         10    15321338.0    22159363.0      786521    22312097    9322731.5  CUDA_KERNEL  add_threadBlockBF16_kernel(int, const __nv_bfloat16 *, __nv_bfloat16 *)
      0.3        102423509         10    10242350.9    10232088.0     1673298    18828414    9025949.0  CUDA_KERNEL  add_threadBlock_kernel(int, const float *, float *)                    
```

#### Laptop (RTX 4060 laptop), on battery

```text
Time (%)  Total Time (ns)  Instances   Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)   Category     Operation
 --------  ---------------  ---------  ----------  ----------  --------  --------  -----------  -----------  -----------------------------------------------------------------------
     99.2        663005681         10  66300568.1  66421328.0  65172362  66468000     397411.8  CUDA_KERNEL  add_naive_kernel(int, const float *, float *)                          
      0.8          5134264         10    513426.4    513500.0    511228    515612       1063.7  CUDA_KERNEL  add_block_kernel(int, const float *, float *)                          
      0.0           132223         10     13222.3     13216.0     13183     13248         25.4  CUDA_KERNEL  add_threadBlockBF16_kernel(int, const __nv_bfloat16 *, __nv_bfloat16 *)
      0.0           123871         10     12387.1     12384.0     12383     12416         10.2  CUDA_KERNEL  add_threadBlock_kernel(int, const float *, float *)
```