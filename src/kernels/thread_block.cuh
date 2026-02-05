#pragma once

#include "../data.h"
#include "../bench.h"
#include <cuda_runtime.h>
#include <cstddef>

#define DECL(base)                                                                   \
    float base##mo(const data::DeviceData<float>& d, cudaStream_t s, int blockSize); \
    float base##gs(const data::DeviceData<float>& d, cudaStream_t s, int blockSize, int numBlocks);

// V1U1 => N
DECL(addThreadBlockIV1U1NC)
DECL(addThreadBlockIV1U1NR)
DECL(addThreadBlockSV1U1NC)
DECL(addThreadBlockSV1U1NR)

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
    DECL(addThreadBlock##IndexTag##VTag##UTag##ModeTag##CRTag)

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

// - gridStrideBlocks only used by grid-stride variants
inline void benchAllThreadBlock(const std::size_t n,
                                const int repeats,
                                cudaStream_t s,
                                const int blockSize = 256,
                                const int gridStrideBlocks = 2048,
                                const bool onlyRestrict = true) {
    #define BENCH_ONE_MO(base, label) \
        bench(label, n, repeats, s, [=](const data::DeviceData<float>& d, cudaStream_t st) { \
            return base##mo(d, st, blockSize); \
        });

    #define BENCH_ONE_GS(base, label) \
        bench(label, n, repeats, s, [=](const data::DeviceData<float>& d, cudaStream_t st) { \
            return base##gs(d, st, blockSize, gridStrideBlocks); \
        });

    #define BENCH_N(IndexTag, CRTag) do { \
        BENCH_ONE_MO(addThreadBlock##IndexTag##V1U1N##CRTag, "addThreadBlock" #IndexTag "V1U1N" #CRTag "mo (FP32)") \
        BENCH_ONE_GS(addThreadBlock##IndexTag##V1U1N##CRTag, "addThreadBlock" #IndexTag "V1U1N" #CRTag "gs (FP32)") \
    } while (0);

    if (onlyRestrict) {
        BENCH_N(I, R)
        BENCH_N(S, R)
    } else {
        BENCH_N(I, C)
        BENCH_N(I, R)
        BENCH_N(S, C)
        BENCH_N(S, R)
    }
    #undef BENCH_N

    #define BENCH_TM(IndexTag, VTag, UTag, ModeTag, CRTag) do { \
        BENCH_ONE_MO(addThreadBlock##IndexTag##VTag##UTag##ModeTag##CRTag, \
        "addThreadBlock" #IndexTag #VTag #UTag #ModeTag #CRTag "mo (FP32)") \
        BENCH_ONE_GS(addThreadBlock##IndexTag##VTag##UTag##ModeTag##CRTag, \
        "addThreadBlock" #IndexTag #VTag #UTag #ModeTag #CRTag "gs (FP32)") \
    } while (0);

    #define BENCH_FOR_ONE_VU(VTag, VecT, Shift, UTag, U) do { \
        if (onlyRestrict) { \
            BENCH_TM(I, VTag, UTag, T, R) \
            BENCH_TM(I, VTag, UTag, M, R) \
            BENCH_TM(S, VTag, UTag, T, R) \
            BENCH_TM(S, VTag, UTag, M, R) \
        } else { \
            BENCH_TM(I, VTag, UTag, T, C) \
            BENCH_TM(I, VTag, UTag, T, R) \
            BENCH_TM(I, VTag, UTag, M, C) \
            BENCH_TM(I, VTag, UTag, M, R) \
            BENCH_TM(S, VTag, UTag, T, C) \
            BENCH_TM(S, VTag, UTag, T, R) \
            BENCH_TM(S, VTag, UTag, M, C) \
            BENCH_TM(S, VTag, UTag, M, R) \
        } \
    } while (0);

    VU_LIST_EXCEPT_V1U1(BENCH_FOR_ONE_VU)

    #undef BENCH_FOR_ONE_VU
    #undef BENCH_TM
    #undef BENCH_ONE_GS
    #undef BENCH_ONE_MO
}

#undef DECL_FOR_ONE_VU
#undef DECL_TM
#undef VU_LIST_EXCEPT_V1U1
#undef DECL