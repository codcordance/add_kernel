# Add kernel experiments

Some experiments on writing and running CUDA kernels in C++.
The kernel is a really basic one consisting in adding two vectors
of length .

### Results

I run the experiments with $n = 2^{20} = 1 048 576$ and $\texttt{blockSize} = 256$.

#### Laptop (RTX 4060 laptop), plugged in (AC power)

```text
Time (%)  Total Time (ns)  Instances   Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)   Category     Operation
 --------  ---------------  ---------  ----------  ----------  --------  --------  -----------  -----------  -----------------------------------------------------------------------
     99.2        631938433         10  63193843.3  62445858.5  62098358  65148145    1305832.3  CUDA_KERNEL  add_naive_kernel(int, const float *, float *)                          
      0.8          4912248         10    491224.8    491164.0    490300    492476        663.3  CUDA_KERNEL  add_block_kernel(int, const float *, float *)                          
      0.0           131423         10     13142.3     13152.0     13119     13152         15.6  CUDA_KERNEL  add_threadBlockBF16_kernel(int, const __nv_bfloat16 *, __nv_bfloat16 *)
      0.0           109183         10     10918.3     10912.0     10880     10944         20.1  CUDA_KERNEL  add_threadBlock_kernel(int, const float *, float *)
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