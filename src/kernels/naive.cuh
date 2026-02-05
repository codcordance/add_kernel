#pragma once

#include "../data.h"
#include "../bench.h"
#include <cuda_runtime.h>

#define DECL(name) float name(const data::DeviceData<float>& d, cudaStream_t s);

// V1U1 => N
DECL(addNaiveIV1U1NC)
DECL(addNaiveIV1U1NR)
DECL(addNaiveSV1U1NC)
DECL(addNaiveSV1U1NR)

#define VU_LIST_EXCEPT_V1U1(X) \
    X(V1, float , 0, U2, 2)    \
    X(V1, float , 0, U4, 4)    \
    X(V2, float2, 1, U1, 1)    \
    X(V2, float2, 1, U2, 2)    \
    X(V2, float2, 1, U4, 4)    \
    X(V4, float4, 2, U1, 1)    \
    X(V4, float4, 2, U2, 2)    \
    X(V4, float4, 2, U4, 4)

#define DECL_TM(IndexTag, VTag, UTag, ModeTag, CRTag) \
    DECL(addNaive##IndexTag##VTag##UTag##ModeTag##CRTag)

#define DECL_FOR_ONE_VU(VTag, VecT, Shift, UTag, U) \
    DECL_TM(I, VTag, UTag, T, C)                    \
    DECL_TM(I, VTag, UTag, T, R)                    \
    DECL_TM(I, VTag, UTag, M, C)                    \
    DECL_TM(I, VTag, UTag, M, R)                    \
    DECL_TM(S, VTag, UTag, T, C)                    \
    DECL_TM(S, VTag, UTag, T, R)                    \
    DECL_TM(S, VTag, UTag, M, C)                    \
    DECL_TM(S, VTag, UTag, M, R)

VU_LIST_EXCEPT_V1U1(DECL_FOR_ONE_VU)

inline void benchAllNaive(const std::size_t n, const int repeats, cudaStream_t s) {
    #define BENCH_N(IndexTag, CRTag) \
        bench("addNaive" #IndexTag "V1U1N" #CRTag " (FP32)", n, repeats, s, addNaive##IndexTag##V1U1N##CRTag);

    BENCH_N(I, C)
    BENCH_N(I, R)
    BENCH_N(S, C)
    BENCH_N(S, R)

    #undef BENCH_N

    #define BENCH_ONE(IndexTag, VTag, UTag, ModeTag, CRTag) \
        bench("addNaive" #IndexTag #VTag #UTag #ModeTag #CRTag " (FP32)", n, repeats, s, addNaive##IndexTag##VTag##UTag##ModeTag##CRTag);

    #define BENCH_FOR_ONE_VU(VTag, VecT, Shift, UTag, U) \
        BENCH_ONE(I, VTag, UTag, T, C)                   \
        BENCH_ONE(I, VTag, UTag, T, R)                   \
        BENCH_ONE(I, VTag, UTag, M, C)                   \
        BENCH_ONE(I, VTag, UTag, M, R)                   \
        BENCH_ONE(S, VTag, UTag, T, C)                   \
        BENCH_ONE(S, VTag, UTag, T, R)                   \
        BENCH_ONE(S, VTag, UTag, M, C)                   \
        BENCH_ONE(S, VTag, UTag, M, R)

    VU_LIST_EXCEPT_V1U1(BENCH_FOR_ONE_VU)

    #undef BENCH_FOR_ONE_VU
    #undef BENCH_ONE
}

#undef DECL_FOR_ONE_VU
#undef DECL_TM
#undef VU_LIST_EXCEPT_V1U1
#undef DECL
