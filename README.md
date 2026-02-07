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
- $s$ is the format size: $s = 4\ \text B$ for `float`/FP32 (it would be $s = 2\ \text B$ for `__nv_bfloat16`/BF16)

### Hardware & Theoretical bandwidth

The experiments were run on my laptop, with a _NVIDIA Geforce RTX 4060 Laptop_ GPU. According to
[NotebookCheck.net](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4060-Laptop-GPU-Benchmarks-and-Specs.675692.0.html),
it has a clock speed of 16 Gbps (effective) and a 128-Bit memory bus, so the theoretical bandwidth is 
$$16 \times 128 / 8 = 256\ \text{GB/s}$$

### Kernels

My results are based on the following configuration:

* $n = 2^{27} = 134,217,728$ (an array of $n$ `float`/FP32 is $\approx 0.54\ \text{GB}$) ;
* each kernel is run $r = 4$ times ;
* laptop power supply connected (kernel seems to run slower when running on battery).

The benchmark covers two families of kernels: a naive implementation (one thread does all the work) and a thread-block implementation (work is distributed across threads, with a grid-wide traversal of the arrays). All kernels implement the same elementwise addition and differ only by indexing choices, pointer qualifiers, and memory-access patterns.

I use the following abbreviations in kernel names:

- **C vs R** indicates whether pointers are passed as `__restrict__` (`C` for baseline pointers, `R`: for `__restrict__` pointers, i.e., non-aliasing assumption for the compiler).

    In all cases, using `__restrict__` is beneficial in my measurements (monotone improvement), so **all thread-block kernels are benchmarked in the `R` configuration**.

- **I vs S** indicates the type used for loop counters / indices: `I` for indexing with signed `int` (see [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#loop-counters-signed-vs-unsigned)
on Loop Counters Signed vs. Unsigned), `S` for indexing with `size_t` (unsigned).

- **V** and **U** encode vector width and unrolling:

  * `V1`, `V2`, `V4`: scalar (`float`), 2-wide, or 4-wide vectorized operations
  * `U1`, `U2`, `U4`: unroll factor in the main loop body

- **N / T / M** encode how vectorization is expressed:

  * `N`: “no vectorization” baseline; this exists only for the `V1U1` case (pure scalar loads/stores).
  * `T`: vectorized memory operations while keeping the API as `float*` (vector loads/stores generated from `float*` addresses).
  * `M`: vector pointer API (`float2*` / `float4*`), i.e., vectorization is expressed in the pointer type.

- For thread-block kernels, **gs** and **mo** use two traversal strategies:

  * grid-stride `gs`: grid-stride loop to cover the full array. These kernels were run with multiple launch configurations (different block sizes and numbers of blocks).
  * monolithic `mo`: enough threads to cover the whole array (no loop). Run with variable block sizes and `(n + blockSize - 1) / blockSize` blocks.


## Results

`__restrict__` yields a consistent, monotone gain in my measurements. Based on this, all thread-block results reported below use `__restrict__`.

The best performing kernel is `addThreadBlockIV4U4MRmo` at $\approx 241\ \text{GB/s}$.
Within the thread-block kernels, there is a reproducible spread (roughly $230 \pm 10\ \text{GB/s}$ effective bandwidth) even when restricting attention to a single access mode (e.g., only `M` kernels or only `T` kernels). Some effects I observed are:
1. grid-stride vs monolithic: For some configurations, the `gs` variant is measurably slower than the `mo` one. This effect is not uniform across all kernels: it is most visible for `T` kernels (vector loads/stores derived from `float*`), and typically smaller for `M` kernels (vector pointer API). A plausible explanation is that the traversal changes the distribution of effective alignment and boundary-crossing patterns at the warp level. In `T`, because the base pointers are `float*`, alignment guarantees are weaker at the type level; generating 16-byte vector transactions from such addresses can be more sensitive to how indices map to byte addresses. In contrast, `M` communicates alignment through the pointer type and tends to behave more consistently across traversals.

2. Unrolling helps (`U1` < `U2` < `U4`). Increasing `U` tends to increase achieved bandwidth (probably since it increases ILP (instruction-level parallelism) and keeps more memory operations in flight, hiding latency). In my results, `U4` is very often near the best.

3. Vectorization helps (`V1` < `V2` < `V4`, when accesses stay well-behaved).
When alignment and coalescing are good, `V4` variants are typically at or near the top.

4. Following NVIDIA recommendations, indexing with `int` (`I` kernels) perform slightly better than `size_t` (`S` kernels) for most kernels.

As an illustration of the `T` vs `M` distinction, the `T` kernels typically contain PTX sequences of the form:

```ptx
ld.global.nc.v4.f32  {...}, [addr_from_float_ptr];
ld.global.v4.f32     {...}, [addr_from_float_ptr];
st.global.v4.f32     [addr_from_float_ptr], {...};
```

while the `M` kernels correspond to a `float4*` API and tend to map more directly to 128-bit memory instructions in SASS (e.g., `LDG.E.128` / `STG.E.128` patterns), with less sensitivity to how vectorization is "reconstructed" from scalar pointer arithmetic.

**Note on register count**: Register count correlates with performance in a few cases but it is not sufficient by itself to explain all variations. For instance, in the V4U4 grid-stride pair shown below, the M-variant uses more registers than the T-variant (48 vs 39), which can reduce occupancy; this is a reasonable contributing factor to the observed throughput gap, but the final performance also depends on instruction mix and scheduling (e.g., the balance between integer address-generation and vector memory operations).

---

### Table of results

#### naive implementation (all work in one thread)

```text
Name                                       Mean duration |  Mean bandwidth  |    Total error
============================================================================================
addNaiveIV1U1NC (FP32)                     11527.752 ms  |       0.14 GB/s  |      0.000e+00
addNaiveIV1U1NR (FP32)                      3215.736 ms  |       0.50 GB/s  |      0.000e+00
addNaiveSV1U1NC (FP32)                      6459.632 ms  |       0.25 GB/s  |      0.000e+00
addNaiveSV1U1NR (FP32)                      5113.176 ms  |       0.31 GB/s  |      0.000e+00
addNaiveIV1U2TC (FP32)                      6454.903 ms  |       0.25 GB/s  |      0.000e+00
addNaiveIV1U2TR (FP32)                      4672.858 ms  |       0.34 GB/s  |      0.000e+00
addNaiveIV1U2MC (FP32)                      6454.810 ms  |       0.25 GB/s  |      0.000e+00
addNaiveIV1U2MR (FP32)                      4672.993 ms  |       0.34 GB/s  |      0.000e+00
addNaiveSV1U2TC (FP32)                      6448.609 ms  |       0.25 GB/s  |      0.000e+00
addNaiveSV1U2TR (FP32)                      4684.689 ms  |       0.34 GB/s  |      0.000e+00
addNaiveSV1U2MC (FP32)                      6454.169 ms  |       0.25 GB/s  |      0.000e+00
addNaiveSV1U2MR (FP32)                      4770.141 ms  |       0.34 GB/s  |      0.000e+00
addNaiveIV1U4TC (FP32)                     11517.314 ms  |       0.14 GB/s  |      0.000e+00
addNaiveIV1U4TR (FP32)                      3584.026 ms  |       0.45 GB/s  |      0.000e+00
addNaiveIV1U4MC (FP32)                     11515.562 ms  |       0.14 GB/s  |      0.000e+00
addNaiveIV1U4MR (FP32)                      3212.648 ms  |       0.50 GB/s  |      0.000e+00
addNaiveSV1U4TC (FP32)                     11260.140 ms  |       0.14 GB/s  |      0.000e+00
addNaiveSV1U4TR (FP32)                      3281.683 ms  |       0.49 GB/s  |      0.000e+00
addNaiveSV1U4MC (FP32)                     11282.765 ms  |       0.14 GB/s  |      0.000e+00
addNaiveSV1U4MR (FP32)                      3237.980 ms  |       0.50 GB/s  |      0.000e+00
addNaiveIV2U1TC (FP32)                      8415.338 ms  |       0.19 GB/s  |      0.000e+00
addNaiveIV2U1TR (FP32)                      3554.149 ms  |       0.45 GB/s  |      0.000e+00
addNaiveIV2U1MC (FP32)                      8413.814 ms  |       0.19 GB/s  |      0.000e+00
addNaiveIV2U1MR (FP32)                      4106.736 ms  |       0.39 GB/s  |      0.000e+00
addNaiveSV2U1TC (FP32)                      5370.547 ms  |       0.30 GB/s  |      0.000e+00
addNaiveSV2U1TR (FP32)                      4630.392 ms  |       0.35 GB/s  |      0.000e+00
addNaiveSV2U1MC (FP32)                      5370.587 ms  |       0.30 GB/s  |      0.000e+00
addNaiveSV2U1MR (FP32)                      4598.255 ms  |       0.35 GB/s  |      0.000e+00
addNaiveIV2U2TC (FP32)                      7419.546 ms  |       0.22 GB/s  |      0.000e+00
addNaiveIV2U2TR (FP32)                      3208.450 ms  |       0.50 GB/s  |      0.000e+00
addNaiveIV2U2MC (FP32)                      7419.702 ms  |       0.22 GB/s  |      0.000e+00
addNaiveIV2U2MR (FP32)                      2747.487 ms  |       0.59 GB/s  |      0.000e+00
addNaiveSV2U2TC (FP32)                      7419.953 ms  |       0.22 GB/s  |      0.000e+00
addNaiveSV2U2TR (FP32)                      4550.607 ms  |       0.35 GB/s  |      0.000e+00
addNaiveSV2U2MC (FP32)                      7439.032 ms  |       0.22 GB/s  |      0.000e+00
addNaiveSV2U2MR (FP32)                      2803.829 ms  |       0.57 GB/s  |      0.000e+00
addNaiveIV2U4TC (FP32)                      8392.718 ms  |       0.19 GB/s  |      0.000e+00
addNaiveIV2U4TR (FP32)                      3398.015 ms  |       0.47 GB/s  |      0.000e+00
addNaiveIV2U4MC (FP32)                      8393.080 ms  |       0.19 GB/s  |      0.000e+00
addNaiveIV2U4MR (FP32)                      4126.830 ms  |       0.39 GB/s  |      0.000e+00
addNaiveSV2U4TC (FP32)                      8393.038 ms  |       0.19 GB/s  |      0.000e+00
addNaiveSV2U4TR (FP32)                      3597.978 ms  |       0.45 GB/s  |      0.000e+00
addNaiveSV2U4MC (FP32)                      8413.618 ms  |       0.19 GB/s  |      0.000e+00
addNaiveSV2U4MR (FP32)                      3549.826 ms  |       0.45 GB/s  |      0.000e+00
addNaiveIV4U1TC (FP32)                      5614.123 ms  |       0.29 GB/s  |      0.000e+00
addNaiveIV4U1TR (FP32)                      3439.715 ms  |       0.47 GB/s  |      0.000e+00
addNaiveIV4U1MC (FP32)                      5614.056 ms  |       0.29 GB/s  |      0.000e+00
addNaiveIV4U1MR (FP32)                      3345.624 ms  |       0.48 GB/s  |      0.000e+00
addNaiveSV4U1TC (FP32)                      4750.802 ms  |       0.34 GB/s  |      0.000e+00
addNaiveSV4U1TR (FP32)                      2688.036 ms  |       0.60 GB/s  |      0.000e+00
addNaiveSV4U1MC (FP32)                      4691.486 ms  |       0.34 GB/s  |      0.000e+00
addNaiveSV4U1MR (FP32)                      2553.009 ms  |       0.63 GB/s  |      0.000e+00
addNaiveIV4U2TC (FP32)                      4676.346 ms  |       0.34 GB/s  |      0.000e+00
addNaiveIV4U2TR (FP32)                      3479.163 ms  |       0.46 GB/s  |      0.000e+00
addNaiveIV4U2MC (FP32)                      5388.808 ms  |       0.30 GB/s  |      0.000e+00
addNaiveIV4U2MR (FP32)                      2963.470 ms  |       0.54 GB/s  |      0.000e+00
addNaiveSV4U2TC (FP32)                      5796.180 ms  |       0.28 GB/s  |      0.000e+00
addNaiveSV4U2TR (FP32)                      4409.209 ms  |       0.37 GB/s  |      0.000e+00
addNaiveSV4U2MC (FP32)                      5825.625 ms  |       0.28 GB/s  |      0.000e+00
addNaiveSV4U2MR (FP32)                      3466.583 ms  |       0.46 GB/s  |      0.000e+00
addNaiveIV4U4TC (FP32)                      4676.416 ms  |       0.34 GB/s  |      0.000e+00
addNaiveIV4U4TR (FP32)                      3479.116 ms  |       0.46 GB/s  |      0.000e+00
addNaiveIV4U4MC (FP32)                      5388.819 ms  |       0.30 GB/s  |      0.000e+00
addNaiveIV4U4MR (FP32)                      2963.362 ms  |       0.54 GB/s  |      0.000e+00
addNaiveSV4U4TC (FP32)                      5796.208 ms  |       0.28 GB/s  |      0.000e+00
addNaiveSV4U4TR (FP32)                      4306.069 ms  |       0.37 GB/s  |      0.000e+00
addNaiveSV4U4MC (FP32)                      5790.847 ms  |       0.28 GB/s  |      0.000e+00
addNaiveSV4U4MR (FP32)                      2963.809 ms  |       0.54 GB/s  |      0.000e+00
```

#### blockSize = 256, gridStrideBlocks = 2048

```text
Name                                       Mean duration |  Mean bandwidth  |    Total error
============================================================================================
addThreadBlockIV1U1NRmo (FP32)                 7.125 ms  |     226.06 GB/s  |      0.000e+00
addThreadBlockIV1U1NRgs (FP32)                 7.052 ms  |     228.40 GB/s  |      0.000e+00
addThreadBlockSV1U1NRmo (FP32)                 7.118 ms  |     226.28 GB/s  |      0.000e+00
addThreadBlockSV1U1NRgs (FP32)                 7.059 ms  |     228.17 GB/s  |      0.000e+00
addThreadBlockIV1U2TRmo (FP32)                 7.042 ms  |     228.73 GB/s  |      0.000e+00
addThreadBlockIV1U2TRgs (FP32)                 7.080 ms  |     227.49 GB/s  |      0.000e+00
addThreadBlockIV1U2MRmo (FP32)                 7.080 ms  |     227.49 GB/s  |      0.000e+00
addThreadBlockIV1U2MRgs (FP32)                 7.057 ms  |     228.24 GB/s  |      0.000e+00
addThreadBlockSV1U2TRmo (FP32)                 7.066 ms  |     227.94 GB/s  |      0.000e+00
addThreadBlockSV1U2TRgs (FP32)                 7.059 ms  |     228.15 GB/s  |      0.000e+00
addThreadBlockSV1U2MRmo (FP32)                 7.038 ms  |     228.84 GB/s  |      0.000e+00
addThreadBlockSV1U2MRgs (FP32)                 7.045 ms  |     228.61 GB/s  |      0.000e+00
addThreadBlockIV1U4TRmo (FP32)                 6.912 ms  |     233.03 GB/s  |      0.000e+00
addThreadBlockIV1U4TRgs (FP32)                 6.856 ms  |     234.91 GB/s  |      0.000e+00
addThreadBlockIV1U4MRmo (FP32)                 6.899 ms  |     233.47 GB/s  |      0.000e+00
addThreadBlockIV1U4MRgs (FP32)                 6.854 ms  |     234.98 GB/s  |      0.000e+00
addThreadBlockSV1U4TRmo (FP32)                 6.906 ms  |     233.22 GB/s  |      0.000e+00
addThreadBlockSV1U4TRgs (FP32)                 6.848 ms  |     235.18 GB/s  |      0.000e+00
addThreadBlockSV1U4MRmo (FP32)                 6.911 ms  |     233.03 GB/s  |      0.000e+00
addThreadBlockSV1U4MRgs (FP32)                 6.838 ms  |     235.53 GB/s  |      0.000e+00
addThreadBlockIV2U1TRmo (FP32)                 7.064 ms  |     228.02 GB/s  |      0.000e+00
addThreadBlockIV2U1TRgs (FP32)                 7.113 ms  |     226.44 GB/s  |      0.000e+00
addThreadBlockIV2U1MRmo (FP32)                 7.046 ms  |     228.59 GB/s  |      0.000e+00
addThreadBlockIV2U1MRgs (FP32)                 7.074 ms  |     227.67 GB/s  |      0.000e+00
addThreadBlockSV2U1TRmo (FP32)                 7.074 ms  |     227.68 GB/s  |      0.000e+00
addThreadBlockSV2U1TRgs (FP32)                 7.304 ms  |     220.52 GB/s  |      0.000e+00
addThreadBlockSV2U1MRmo (FP32)                 7.047 ms  |     228.55 GB/s  |      0.000e+00
addThreadBlockSV2U1MRgs (FP32)                 7.104 ms  |     226.73 GB/s  |      0.000e+00
addThreadBlockIV2U2TRmo (FP32)                 6.873 ms  |     234.33 GB/s  |      0.000e+00
addThreadBlockIV2U2TRgs (FP32)                 6.870 ms  |     234.44 GB/s  |      0.000e+00
addThreadBlockIV2U2MRmo (FP32)                 6.911 ms  |     233.06 GB/s  |      0.000e+00
addThreadBlockIV2U2MRgs (FP32)                 6.875 ms  |     234.26 GB/s  |      0.000e+00
addThreadBlockSV2U2TRmo (FP32)                 6.889 ms  |     233.79 GB/s  |      0.000e+00
addThreadBlockSV2U2TRgs (FP32)                 6.891 ms  |     233.73 GB/s  |      0.000e+00
addThreadBlockSV2U2MRmo (FP32)                 6.881 ms  |     234.05 GB/s  |      0.000e+00
addThreadBlockSV2U2MRgs (FP32)                 6.883 ms  |     234.01 GB/s  |      0.000e+00
addThreadBlockIV2U4TRmo (FP32)                 6.761 ms  |     238.22 GB/s  |      0.000e+00
addThreadBlockIV2U4TRgs (FP32)                 6.749 ms  |     238.66 GB/s  |      0.000e+00
addThreadBlockIV2U4MRmo (FP32)                 6.758 ms  |     238.32 GB/s  |      0.000e+00
addThreadBlockIV2U4MRgs (FP32)                 6.743 ms  |     238.87 GB/s  |      0.000e+00
addThreadBlockSV2U4TRmo (FP32)                 6.829 ms  |     235.86 GB/s  |      0.000e+00
addThreadBlockSV2U4TRgs (FP32)                 6.796 ms  |     237.01 GB/s  |      0.000e+00
addThreadBlockSV2U4MRmo (FP32)                 6.792 ms  |     237.14 GB/s  |      0.000e+00
addThreadBlockSV2U4MRgs (FP32)                 6.781 ms  |     237.52 GB/s  |      0.000e+00
addThreadBlockIV4U1TRmo (FP32)                 6.890 ms  |     233.75 GB/s  |      0.000e+00
addThreadBlockIV4U1TRgs (FP32)                 6.886 ms  |     233.91 GB/s  |      0.000e+00
addThreadBlockIV4U1MRmo (FP32)                 6.936 ms  |     232.20 GB/s  |      0.000e+00
addThreadBlockIV4U1MRgs (FP32)                 6.865 ms  |     234.62 GB/s  |      0.000e+00
addThreadBlockSV4U1TRmo (FP32)                 6.907 ms  |     233.19 GB/s  |      0.000e+00
addThreadBlockSV4U1TRgs (FP32)                 6.927 ms  |     232.51 GB/s  |      0.000e+00
addThreadBlockSV4U1MRmo (FP32)                 6.930 ms  |     232.43 GB/s  |      0.000e+00
addThreadBlockSV4U1MRgs (FP32)                 6.883 ms  |     233.98 GB/s  |      0.000e+00
addThreadBlockIV4U2TRmo (FP32)                 6.943 ms  |     231.98 GB/s  |      0.000e+00
addThreadBlockIV4U2TRgs (FP32)                 6.851 ms  |     235.09 GB/s  |      0.000e+00
addThreadBlockIV4U2MRmo (FP32)                 6.950 ms  |     231.75 GB/s  |      0.000e+00
addThreadBlockIV4U2MRgs (FP32)                 6.850 ms  |     235.12 GB/s  |      0.000e+00
addThreadBlockSV4U2TRmo (FP32)                 6.927 ms  |     232.50 GB/s  |      0.000e+00
addThreadBlockSV4U2TRgs (FP32)                 6.867 ms  |     234.56 GB/s  |      0.000e+00
addThreadBlockSV4U2MRmo (FP32)                 6.922 ms  |     232.69 GB/s  |      0.000e+00
addThreadBlockSV4U2MRgs (FP32)                 6.846 ms  |     235.26 GB/s  |      0.000e+00
addThreadBlockIV4U4TRmo (FP32)                 6.882 ms  |     234.03 GB/s  |      0.000e+00
addThreadBlockIV4U4TRgs (FP32)                 7.206 ms  |     223.53 GB/s  |      0.000e+00
addThreadBlockIV4U4MRmo (FP32)                 6.735 ms  |     239.15 GB/s  |      0.000e+00
addThreadBlockIV4U4MRgs (FP32)                 6.797 ms  |     236.96 GB/s  |      0.000e+00
addThreadBlockSV4U4TRmo (FP32)                 6.857 ms  |     234.87 GB/s  |      0.000e+00
addThreadBlockSV4U4TRgs (FP32)                 6.826 ms  |     235.94 GB/s  |      0.000e+00
addThreadBlockSV4U4MRmo (FP32)                 6.815 ms  |     236.34 GB/s  |      0.000e+00
addThreadBlockSV4U4MRgs (FP32)                 6.798 ms  |     236.91 GB/s  |      0.000e+00
```

#### blockSize = 384, gridStrideBlocks = 2048

```text
Name                                       Mean duration |  Mean bandwidth  |    Total error
============================================================================================
addThreadBlockIV1U1NRmo (FP32)                 7.088 ms  |     227.23 GB/s  |      0.000e+00
addThreadBlockIV1U1NRgs (FP32)                 7.091 ms  |     227.15 GB/s  |      0.000e+00
addThreadBlockSV1U1NRmo (FP32)                 7.136 ms  |     225.70 GB/s  |      0.000e+00
addThreadBlockSV1U1NRgs (FP32)                 7.034 ms  |     228.98 GB/s  |      0.000e+00
addThreadBlockIV1U2TRmo (FP32)                 7.060 ms  |     228.12 GB/s  |      0.000e+00
addThreadBlockIV1U2TRgs (FP32)                 7.067 ms  |     227.92 GB/s  |      0.000e+00
addThreadBlockIV1U2MRmo (FP32)                 7.069 ms  |     227.84 GB/s  |      0.000e+00
addThreadBlockIV1U2MRgs (FP32)                 7.024 ms  |     229.31 GB/s  |      0.000e+00
addThreadBlockSV1U2TRmo (FP32)                 7.053 ms  |     228.37 GB/s  |      0.000e+00
addThreadBlockSV1U2TRgs (FP32)                 7.046 ms  |     228.57 GB/s  |      0.000e+00
addThreadBlockSV1U2MRmo (FP32)                 7.061 ms  |     228.08 GB/s  |      0.000e+00
addThreadBlockSV1U2MRgs (FP32)                 7.075 ms  |     227.64 GB/s  |      0.000e+00
addThreadBlockIV1U4TRmo (FP32)                 6.865 ms  |     234.60 GB/s  |      0.000e+00
addThreadBlockIV1U4TRgs (FP32)                 6.856 ms  |     234.91 GB/s  |      0.000e+00
addThreadBlockIV1U4MRmo (FP32)                 6.861 ms  |     234.74 GB/s  |      0.000e+00
addThreadBlockIV1U4MRgs (FP32)                 6.861 ms  |     234.76 GB/s  |      0.000e+00
addThreadBlockSV1U4TRmo (FP32)                 6.955 ms  |     231.56 GB/s  |      0.000e+00
addThreadBlockSV1U4TRgs (FP32)                 6.821 ms  |     236.13 GB/s  |      0.000e+00
addThreadBlockSV1U4MRmo (FP32)                 6.893 ms  |     233.67 GB/s  |      0.000e+00
addThreadBlockSV1U4MRgs (FP32)                 6.855 ms  |     234.96 GB/s  |      0.000e+00
addThreadBlockIV2U1TRmo (FP32)                 7.062 ms  |     228.06 GB/s  |      0.000e+00
addThreadBlockIV2U1TRgs (FP32)                 7.081 ms  |     227.45 GB/s  |      0.000e+00
addThreadBlockIV2U1MRmo (FP32)                 7.049 ms  |     228.48 GB/s  |      0.000e+00
addThreadBlockIV2U1MRgs (FP32)                 7.085 ms  |     227.32 GB/s  |      0.000e+00
addThreadBlockSV2U1TRmo (FP32)                 7.044 ms  |     228.64 GB/s  |      0.000e+00
addThreadBlockSV2U1TRgs (FP32)                 7.103 ms  |     226.74 GB/s  |      0.000e+00
addThreadBlockSV2U1MRmo (FP32)                 7.034 ms  |     228.97 GB/s  |      0.000e+00
addThreadBlockSV2U1MRgs (FP32)                 7.052 ms  |     228.38 GB/s  |      0.000e+00
addThreadBlockIV2U2TRmo (FP32)                 6.910 ms  |     233.09 GB/s  |      0.000e+00
addThreadBlockIV2U2TRgs (FP32)                 6.852 ms  |     235.07 GB/s  |      0.000e+00
addThreadBlockIV2U2MRmo (FP32)                 6.898 ms  |     233.51 GB/s  |      0.000e+00
addThreadBlockIV2U2MRgs (FP32)                 6.856 ms  |     234.92 GB/s  |      0.000e+00
addThreadBlockSV2U2TRmo (FP32)                 6.896 ms  |     233.55 GB/s  |      0.000e+00
addThreadBlockSV2U2TRgs (FP32)                 6.916 ms  |     232.89 GB/s  |      0.000e+00
addThreadBlockSV2U2MRmo (FP32)                 6.903 ms  |     233.30 GB/s  |      0.000e+00
addThreadBlockSV2U2MRgs (FP32)                 6.869 ms  |     234.46 GB/s  |      0.000e+00
addThreadBlockIV2U4TRmo (FP32)                 6.767 ms  |     238.01 GB/s  |      0.000e+00
addThreadBlockIV2U4TRgs (FP32)                 6.775 ms  |     237.74 GB/s  |      0.000e+00
addThreadBlockIV2U4MRmo (FP32)                 6.763 ms  |     238.14 GB/s  |      0.000e+00
addThreadBlockIV2U4MRgs (FP32)                 6.760 ms  |     238.25 GB/s  |      0.000e+00
addThreadBlockSV2U4TRmo (FP32)                 6.818 ms  |     236.22 GB/s  |      0.000e+00
addThreadBlockSV2U4TRgs (FP32)                 6.755 ms  |     238.42 GB/s  |      0.000e+00
addThreadBlockSV2U4MRmo (FP32)                 6.778 ms  |     237.64 GB/s  |      0.000e+00
addThreadBlockSV2U4MRgs (FP32)                 6.708 ms  |     240.10 GB/s  |      0.000e+00
addThreadBlockIV4U1TRmo (FP32)                 6.947 ms  |     231.85 GB/s  |      0.000e+00
addThreadBlockIV4U1TRgs (FP32)                 6.913 ms  |     232.97 GB/s  |      0.000e+00
addThreadBlockIV4U1MRmo (FP32)                 6.947 ms  |     231.84 GB/s  |      0.000e+00
addThreadBlockIV4U1MRgs (FP32)                 6.935 ms  |     232.23 GB/s  |      0.000e+00
addThreadBlockSV4U1TRmo (FP32)                 6.956 ms  |     231.55 GB/s  |      0.000e+00
addThreadBlockSV4U1TRgs (FP32)                 6.895 ms  |     233.61 GB/s  |      0.000e+00
addThreadBlockSV4U1MRmo (FP32)                 6.900 ms  |     233.43 GB/s  |      0.000e+00
addThreadBlockSV4U1MRgs (FP32)                 6.879 ms  |     234.14 GB/s  |      0.000e+00
addThreadBlockIV4U2TRmo (FP32)                 6.955 ms  |     231.59 GB/s  |      0.000e+00
addThreadBlockIV4U2TRgs (FP32)                 6.852 ms  |     235.06 GB/s  |      0.000e+00
addThreadBlockIV4U2MRmo (FP32)                 6.918 ms  |     232.80 GB/s  |      0.000e+00
addThreadBlockIV4U2MRgs (FP32)                 6.820 ms  |     236.17 GB/s  |      0.000e+00
addThreadBlockSV4U2TRmo (FP32)                 6.965 ms  |     231.25 GB/s  |      0.000e+00
addThreadBlockSV4U2TRgs (FP32)                 6.816 ms  |     236.30 GB/s  |      0.000e+00
addThreadBlockSV4U2MRmo (FP32)                 6.966 ms  |     231.21 GB/s  |      0.000e+00
addThreadBlockSV4U2MRgs (FP32)                 6.823 ms  |     236.06 GB/s  |      0.000e+00
addThreadBlockIV4U4TRmo (FP32)                 6.861 ms  |     234.76 GB/s  |      0.000e+00
addThreadBlockIV4U4TRgs (FP32)                 7.174 ms  |     224.50 GB/s  |      0.000e+00
addThreadBlockIV4U4MRmo (FP32)                 6.779 ms  |     237.58 GB/s  |      0.000e+00
addThreadBlockIV4U4MRgs (FP32)                 6.833 ms  |     235.71 GB/s  |      0.000e+00
addThreadBlockSV4U4TRmo (FP32)                 6.872 ms  |     234.36 GB/s  |      0.000e+00
addThreadBlockSV4U4TRgs (FP32)                 6.785 ms  |     237.38 GB/s  |      0.000e+00
addThreadBlockSV4U4MRmo (FP32)                 6.826 ms  |     235.94 GB/s  |      0.000e+00
addThreadBlockSV4U4MRgs (FP32)                 6.820 ms  |     236.16 GB/s  |      0.000e+00
```

#### blockSize = 256, gridStrideBlocks = 768

```text
Name                                       Mean duration |  Mean bandwidth  |    Total error
============================================================================================
addThreadBlockIV1U1NRmo (FP32)                 7.101 ms  |     226.81 GB/s  |      0.000e+00
addThreadBlockIV1U1NRgs (FP32)                 6.984 ms  |     230.63 GB/s  |      0.000e+00
addThreadBlockSV1U1NRmo (FP32)                 7.072 ms  |     227.74 GB/s  |      0.000e+00
addThreadBlockSV1U1NRgs (FP32)                 6.982 ms  |     230.68 GB/s  |      0.000e+00
addThreadBlockIV1U2TRmo (FP32)                 7.011 ms  |     229.73 GB/s  |      0.000e+00
addThreadBlockIV1U2TRgs (FP32)                 7.065 ms  |     227.99 GB/s  |      0.000e+00
addThreadBlockIV1U2MRmo (FP32)                 7.084 ms  |     227.35 GB/s  |      0.000e+00
addThreadBlockIV1U2MRgs (FP32)                 7.001 ms  |     230.07 GB/s  |      0.000e+00
addThreadBlockSV1U2TRmo (FP32)                 7.046 ms  |     228.58 GB/s  |      0.000e+00
addThreadBlockSV1U2TRgs (FP32)                 7.080 ms  |     227.50 GB/s  |      0.000e+00
addThreadBlockSV1U2MRmo (FP32)                 7.066 ms  |     227.93 GB/s  |      0.000e+00
addThreadBlockSV1U2MRgs (FP32)                 7.102 ms  |     226.77 GB/s  |      0.000e+00
addThreadBlockIV1U4TRmo (FP32)                 6.885 ms  |     233.94 GB/s  |      0.000e+00
addThreadBlockIV1U4TRgs (FP32)                 6.857 ms  |     234.89 GB/s  |      0.000e+00
addThreadBlockIV1U4MRmo (FP32)                 6.904 ms  |     233.29 GB/s  |      0.000e+00
addThreadBlockIV1U4MRgs (FP32)                 6.905 ms  |     233.26 GB/s  |      0.000e+00
addThreadBlockSV1U4TRmo (FP32)                 6.881 ms  |     234.08 GB/s  |      0.000e+00
addThreadBlockSV1U4TRgs (FP32)                 6.843 ms  |     235.38 GB/s  |      0.000e+00
addThreadBlockSV1U4MRmo (FP32)                 6.848 ms  |     235.18 GB/s  |      0.000e+00
addThreadBlockSV1U4MRgs (FP32)                 6.844 ms  |     235.35 GB/s  |      0.000e+00
addThreadBlockIV2U1TRmo (FP32)                 7.074 ms  |     227.68 GB/s  |      0.000e+00
addThreadBlockIV2U1TRgs (FP32)                 7.089 ms  |     227.19 GB/s  |      0.000e+00
addThreadBlockIV2U1MRmo (FP32)                 7.061 ms  |     228.11 GB/s  |      0.000e+00
addThreadBlockIV2U1MRgs (FP32)                 7.100 ms  |     226.85 GB/s  |      0.000e+00
addThreadBlockSV2U1TRmo (FP32)                 7.033 ms  |     228.99 GB/s  |      0.000e+00
addThreadBlockSV2U1TRgs (FP32)                 7.111 ms  |     226.50 GB/s  |      0.000e+00
addThreadBlockSV2U1MRmo (FP32)                 7.056 ms  |     228.25 GB/s  |      0.000e+00
addThreadBlockSV2U1MRgs (FP32)                 7.100 ms  |     226.84 GB/s  |      0.000e+00
addThreadBlockIV2U2TRmo (FP32)                 6.892 ms  |     233.69 GB/s  |      0.000e+00
addThreadBlockIV2U2TRgs (FP32)                 6.859 ms  |     234.81 GB/s  |      0.000e+00
addThreadBlockIV2U2MRmo (FP32)                 6.891 ms  |     233.74 GB/s  |      0.000e+00
addThreadBlockIV2U2MRgs (FP32)                 6.875 ms  |     234.26 GB/s  |      0.000e+00
addThreadBlockSV2U2TRmo (FP32)                 6.927 ms  |     232.52 GB/s  |      0.000e+00
addThreadBlockSV2U2TRgs (FP32)                 6.864 ms  |     234.66 GB/s  |      0.000e+00
addThreadBlockSV2U2MRmo (FP32)                 6.921 ms  |     232.72 GB/s  |      0.000e+00
addThreadBlockSV2U2MRgs (FP32)                 6.883 ms  |     233.98 GB/s  |      0.000e+00
addThreadBlockIV2U4TRmo (FP32)                 6.812 ms  |     236.42 GB/s  |      0.000e+00
addThreadBlockIV2U4TRgs (FP32)                 6.747 ms  |     238.71 GB/s  |      0.000e+00
addThreadBlockIV2U4MRmo (FP32)                 6.766 ms  |     238.06 GB/s  |      0.000e+00
addThreadBlockIV2U4MRgs (FP32)                 6.798 ms  |     236.94 GB/s  |      0.000e+00
addThreadBlockSV2U4TRmo (FP32)                 6.857 ms  |     234.90 GB/s  |      0.000e+00
addThreadBlockSV2U4TRgs (FP32)                 6.771 ms  |     237.88 GB/s  |      0.000e+00
addThreadBlockSV2U4MRmo (FP32)                 6.782 ms  |     237.48 GB/s  |      0.000e+00
addThreadBlockSV2U4MRgs (FP32)                 6.728 ms  |     239.38 GB/s  |      0.000e+00
addThreadBlockIV4U1TRmo (FP32)                 6.936 ms  |     232.21 GB/s  |      0.000e+00
addThreadBlockIV4U1TRgs (FP32)                 6.907 ms  |     233.19 GB/s  |      0.000e+00
addThreadBlockIV4U1MRmo (FP32)                 6.879 ms  |     234.13 GB/s  |      0.000e+00
addThreadBlockIV4U1MRgs (FP32)                 6.917 ms  |     232.85 GB/s  |      0.000e+00
addThreadBlockSV4U1TRmo (FP32)                 6.919 ms  |     232.78 GB/s  |      0.000e+00
addThreadBlockSV4U1TRgs (FP32)                 6.940 ms  |     232.06 GB/s  |      0.000e+00
addThreadBlockSV4U1MRmo (FP32)                 6.947 ms  |     231.85 GB/s  |      0.000e+00
addThreadBlockSV4U1MRgs (FP32)                 6.921 ms  |     232.72 GB/s  |      0.000e+00
addThreadBlockIV4U2TRmo (FP32)                 6.953 ms  |     231.64 GB/s  |      0.000e+00
addThreadBlockIV4U2TRgs (FP32)                 6.808 ms  |     236.59 GB/s  |      0.000e+00
addThreadBlockIV4U2MRmo (FP32)                 6.918 ms  |     232.83 GB/s  |      0.000e+00
addThreadBlockIV4U2MRgs (FP32)                 6.820 ms  |     236.16 GB/s  |      0.000e+00
addThreadBlockSV4U2TRmo (FP32)                 6.943 ms  |     231.99 GB/s  |      0.000e+00
addThreadBlockSV4U2TRgs (FP32)                 6.847 ms  |     235.23 GB/s  |      0.000e+00
addThreadBlockSV4U2MRmo (FP32)                 6.957 ms  |     231.53 GB/s  |      0.000e+00
addThreadBlockSV4U2MRgs (FP32)                 6.856 ms  |     234.92 GB/s  |      0.000e+00
addThreadBlockIV4U4TRmo (FP32)                 6.871 ms  |     234.41 GB/s  |      0.000e+00
addThreadBlockIV4U4TRgs (FP32)                 7.257 ms  |     221.94 GB/s  |      0.000e+00
addThreadBlockIV4U4MRmo (FP32)                 6.683 ms  |     240.99 GB/s  |      0.000e+00
addThreadBlockIV4U4MRgs (FP32)                 6.821 ms  |     236.11 GB/s  |      0.000e+00
addThreadBlockSV4U4TRmo (FP32)                 6.862 ms  |     234.71 GB/s  |      0.000e+00
addThreadBlockSV4U4TRgs (FP32)                 6.795 ms  |     237.04 GB/s  |      0.000e+00
addThreadBlockSV4U4MRmo (FP32)                 6.793 ms  |     237.09 GB/s  |      0.000e+00
addThreadBlockSV4U4MRgs (FP32)                 6.783 ms  |     237.45 GB/s  |      0.000e+00
```

#### blockSize = 384, gridStrideBlocks = 768

```text
Name                                       Mean duration |  Mean bandwidth  |    Total error
============================================================================================
addThreadBlockIV1U1NRmo (FP32)                 7.143 ms  |     225.50 GB/s  |      0.000e+00
addThreadBlockIV1U1NRgs (FP32)                 7.064 ms  |     228.00 GB/s  |      0.000e+00
addThreadBlockSV1U1NRmo (FP32)                 7.097 ms  |     226.96 GB/s  |      0.000e+00
addThreadBlockSV1U1NRgs (FP32)                 7.066 ms  |     227.94 GB/s  |      0.000e+00
addThreadBlockIV1U2TRmo (FP32)                 7.054 ms  |     228.33 GB/s  |      0.000e+00
addThreadBlockIV1U2TRgs (FP32)                 7.061 ms  |     228.09 GB/s  |      0.000e+00
addThreadBlockIV1U2MRmo (FP32)                 7.047 ms  |     228.57 GB/s  |      0.000e+00
addThreadBlockIV1U2MRgs (FP32)                 7.070 ms  |     227.80 GB/s  |      0.000e+00
addThreadBlockSV1U2TRmo (FP32)                 7.039 ms  |     228.82 GB/s  |      0.000e+00
addThreadBlockSV1U2TRgs (FP32)                 7.080 ms  |     227.50 GB/s  |      0.000e+00
addThreadBlockSV1U2MRmo (FP32)                 7.045 ms  |     228.63 GB/s  |      0.000e+00
addThreadBlockSV1U2MRgs (FP32)                 7.096 ms  |     226.97 GB/s  |      0.000e+00
addThreadBlockIV1U4TRmo (FP32)                 6.872 ms  |     234.36 GB/s  |      0.000e+00
addThreadBlockIV1U4TRgs (FP32)                 6.837 ms  |     235.56 GB/s  |      0.000e+00
addThreadBlockIV1U4MRmo (FP32)                 6.889 ms  |     233.78 GB/s  |      0.000e+00
addThreadBlockIV1U4MRgs (FP32)                 6.811 ms  |     236.49 GB/s  |      0.000e+00
addThreadBlockSV1U4TRmo (FP32)                 6.895 ms  |     233.58 GB/s  |      0.000e+00
addThreadBlockSV1U4TRgs (FP32)                 6.875 ms  |     234.27 GB/s  |      0.000e+00
addThreadBlockSV1U4MRmo (FP32)                 6.957 ms  |     231.50 GB/s  |      0.000e+00
addThreadBlockSV1U4MRgs (FP32)                 6.901 ms  |     233.39 GB/s  |      0.000e+00
addThreadBlockIV2U1TRmo (FP32)                 7.089 ms  |     227.20 GB/s  |      0.000e+00
addThreadBlockIV2U1TRgs (FP32)                 7.134 ms  |     225.76 GB/s  |      0.000e+00
addThreadBlockIV2U1MRmo (FP32)                 7.072 ms  |     227.74 GB/s  |      0.000e+00
addThreadBlockIV2U1MRgs (FP32)                 7.139 ms  |     225.61 GB/s  |      0.000e+00
addThreadBlockSV2U1TRmo (FP32)                 7.088 ms  |     227.24 GB/s  |      0.000e+00
addThreadBlockSV2U1TRgs (FP32)                 7.114 ms  |     226.41 GB/s  |      0.000e+00
addThreadBlockSV2U1MRmo (FP32)                 7.069 ms  |     227.84 GB/s  |      0.000e+00
addThreadBlockSV2U1MRgs (FP32)                 7.055 ms  |     228.29 GB/s  |      0.000e+00
addThreadBlockIV2U2TRmo (FP32)                 6.947 ms  |     231.83 GB/s  |      0.000e+00
addThreadBlockIV2U2TRgs (FP32)                 6.843 ms  |     235.35 GB/s  |      0.000e+00
addThreadBlockIV2U2MRmo (FP32)                 6.923 ms  |     232.65 GB/s  |      0.000e+00
addThreadBlockIV2U2MRgs (FP32)                 6.828 ms  |     235.87 GB/s  |      0.000e+00
addThreadBlockSV2U2TRmo (FP32)                 6.898 ms  |     233.49 GB/s  |      0.000e+00
addThreadBlockSV2U2TRgs (FP32)                 6.897 ms  |     233.53 GB/s  |      0.000e+00
addThreadBlockSV2U2MRmo (FP32)                 6.908 ms  |     233.14 GB/s  |      0.000e+00
addThreadBlockSV2U2MRgs (FP32)                 6.839 ms  |     235.51 GB/s  |      0.000e+00
addThreadBlockIV2U4TRmo (FP32)                 6.755 ms  |     238.43 GB/s  |      0.000e+00
addThreadBlockIV2U4TRgs (FP32)                 6.727 ms  |     239.43 GB/s  |      0.000e+00
addThreadBlockIV2U4MRmo (FP32)                 6.808 ms  |     236.56 GB/s  |      0.000e+00
addThreadBlockIV2U4MRgs (FP32)                 6.773 ms  |     237.80 GB/s  |      0.000e+00
addThreadBlockSV2U4TRmo (FP32)                 6.791 ms  |     237.17 GB/s  |      0.000e+00
addThreadBlockSV2U4TRgs (FP32)                 6.762 ms  |     238.20 GB/s  |      0.000e+00
addThreadBlockSV2U4MRmo (FP32)                 6.820 ms  |     236.16 GB/s  |      0.000e+00
addThreadBlockSV2U4MRgs (FP32)                 6.730 ms  |     239.33 GB/s  |      0.000e+00
addThreadBlockIV4U1TRmo (FP32)                 6.970 ms  |     231.07 GB/s  |      0.000e+00
addThreadBlockIV4U1TRgs (FP32)                 6.916 ms  |     232.88 GB/s  |      0.000e+00
addThreadBlockIV4U1MRmo (FP32)                 6.974 ms  |     230.94 GB/s  |      0.000e+00
addThreadBlockIV4U1MRgs (FP32)                 6.924 ms  |     232.61 GB/s  |      0.000e+00
addThreadBlockSV4U1TRmo (FP32)                 6.925 ms  |     232.56 GB/s  |      0.000e+00
addThreadBlockSV4U1TRgs (FP32)                 6.902 ms  |     233.36 GB/s  |      0.000e+00
addThreadBlockSV4U1MRmo (FP32)                 6.905 ms  |     233.25 GB/s  |      0.000e+00
addThreadBlockSV4U1MRgs (FP32)                 6.896 ms  |     233.57 GB/s  |      0.000e+00
addThreadBlockIV4U2TRmo (FP32)                 6.922 ms  |     232.69 GB/s  |      0.000e+00
addThreadBlockIV4U2TRgs (FP32)                 6.847 ms  |     235.21 GB/s  |      0.000e+00
addThreadBlockIV4U2MRmo (FP32)                 6.933 ms  |     232.31 GB/s  |      0.000e+00
addThreadBlockIV4U2MRgs (FP32)                 6.814 ms  |     236.36 GB/s  |      0.000e+00
addThreadBlockSV4U2TRmo (FP32)                 6.921 ms  |     232.70 GB/s  |      0.000e+00
addThreadBlockSV4U2TRgs (FP32)                 6.830 ms  |     235.80 GB/s  |      0.000e+00
addThreadBlockSV4U2MRmo (FP32)                 6.948 ms  |     231.81 GB/s  |      0.000e+00
addThreadBlockSV4U2MRgs (FP32)                 6.840 ms  |     235.46 GB/s  |      0.000e+00
addThreadBlockIV4U4TRmo (FP32)                 6.870 ms  |     234.46 GB/s  |      0.000e+00
addThreadBlockIV4U4TRgs (FP32)                 7.219 ms  |     223.12 GB/s  |      0.000e+00
addThreadBlockIV4U4MRmo (FP32)                 6.764 ms  |     238.11 GB/s  |      0.000e+00
addThreadBlockIV4U4MRgs (FP32)                 6.878 ms  |     234.18 GB/s  |      0.000e+00
addThreadBlockSV4U4TRmo (FP32)                 6.909 ms  |     233.12 GB/s  |      0.000e+00
addThreadBlockSV4U4TRgs (FP32)                 6.841 ms  |     235.45 GB/s  |      0.000e+00
addThreadBlockSV4U4MRmo (FP32)                 6.813 ms  |     236.40 GB/s  |      0.000e+00
addThreadBlockSV4U4MRgs (FP32)                 6.820 ms  |     236.17 GB/s  |      0.000e+00
```
