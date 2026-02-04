#include "naive.cuh"
#include "../cuda_utils.h"

#include <cuda_runtime.h>
#include <cstddef>

// =========================
// Vectorization & unrolling
// =========================

#define ADD2_INPLACE(a,b) do { (b).x += (a).x; (b).y += (a).y; } while (0)
#define ADD4_INPLACE(a,b) do { (b).x += (a).x; (b).y += (a).y; (b).z += (a).z; (b).w += (a).w; } while (0)

// V1 (float*)
#define BODY_V1_U1(i, x, y) do { (y)[(i)] = (x)[(i)] + (y)[(i)]; } while (0)
#define BODY_V1_U2(i, x, y) do { BODY_V1_U1((i), x, y); BODY_V1_U1((i)+1, x, y); } while (0)
#define BODY_V1_U4(i, x, y) do { BODY_V1_U2((i), x, y); BODY_V1_U2((i)+2, x, y); } while (0)

// V2 (float2*)
#define BODY_V2_U1(i, x2, y2) do { float2 a = (x2)[(i)]; float2 b = (y2)[(i)]; ADD2_INPLACE(a,b); (y2)[(i)] = b; } while (0)
#define BODY_V2_U2(i, x2, y2) do { BODY_V2_U1((i), x2, y2); BODY_V2_U1((i)+1, x2, y2); } while (0)
#define BODY_V2_U4(i, x2, y2) do { BODY_V2_U2((i), x2, y2); BODY_V2_U2((i)+2, x2, y2); } while (0)

// V4 (float4*)
#define BODY_V4_U1(i, x4, y4) do { float4 a = (x4)[(i)]; float4 b = (y4)[(i)]; ADD4_INPLACE(a,b); (y4)[(i)] = b; } while (0)
#define BODY_V4_U2(i, x4, y4) do { BODY_V4_U1((i), x4, y4); BODY_V4_U1((i)+1, x4, y4); } while (0)
#define BODY_V4_U4(i, x4, y4) do { BODY_V4_U2((i), x4, y4); BODY_V4_U2((i)+2, x4, y4); } while (0)

// =================
// Kernel generators
// =================

#define XQ_C
#define YQ_C
#define XQ_R __restrict__
#define YQ_R __restrict__

// N (V1U1N)
#define DEFINE_KERNEL_N(IndexTag, IndexT, CRTag, XQ, YQ) \
__global__ void addNaive##IndexTag##V1U1N##CRTag##_kernel(const IndexT n, const float* XQ x, float* YQ y) { \
    for (IndexT i = 0; i < n; ++i) { BODY_V1_U1(i, x, y); } \
}

// nVec = n >> shift (shift in {1, 2, 3}); tail: i from (nVec<<shift) to n.
#define DEFINE_KERNEL_T(IndexTag, IndexT, VTag, VecT, Shift, UTag, U, CRTag, XQ, YQ)                                \
__global__ void addNaive##IndexTag##VTag##UTag##T##CRTag##_kernel(const IndexT n, const float* XQ x, float* YQ y) { \
    const VecT* XQ xV = (const VecT*)x;                                                                             \
    VecT* YQ yV = (VecT*)y;                                                                                         \
    const IndexT nVec = (IndexT)(n >> (Shift));                                                                     \
    const IndexT mainEnd = (nVec / (IndexT)(U)) * (IndexT)(U);                                                      \
                                                                                                                    \
    for (IndexT i = 0; i < mainEnd; i += (IndexT)(U)) {                                                             \
        BODY_##VTag##_##UTag(i, xV, yV);                                                                            \
    }                                                                                                               \
                                                                                                                    \
    for (IndexT i = mainEnd; i < nVec; ++i) {                                                                       \
        BODY_##VTag##_U1(i, xV, yV);                                                                                \
    }                                                                                                               \
                                                                                                                    \
    for (IndexT i = (IndexT)(nVec << (Shift)); i < n; ++i) {                                                        \
        BODY_V1_U1(i, x, y);                                                                                        \
    }                                                                                                               \
}

#define DEFINE_KERNEL_M(IndexTag, IndexT, VTag, VecT, UTag, U, CRTag, XQ, YQ)                                           \
__global__ void addNaive##IndexTag##VTag##UTag##M##CRTag##_kernel(const IndexT nVec, const VecT* XQ xV, VecT* YQ yV) {  \
    for (IndexT i = 0; i < nVec; i += (IndexT)(U)) {                                                                    \
        BODY_##VTag##_##UTag(i, xV, yV);                                                                                \
    }                                                                                                                   \
}

// ===========================
// Kernel definition w/ tables
// ===========================

DEFINE_KERNEL_N(I, int,    C, XQ_C, YQ_C)
DEFINE_KERNEL_N(I, int,    R, XQ_R, YQ_R)
DEFINE_KERNEL_N(S, size_t, C, XQ_C, YQ_C)
DEFINE_KERNEL_N(S, size_t, R, XQ_R, YQ_R)

// (V,U) tuples except V1U1
#define VU_LIST_EXCEPT_V1U1(X) \
    X(V1, float , 0, U2, 2)    \
    X(V1, float , 0, U4, 4)    \
    X(V2, float2, 1, U1, 1)    \
    X(V2, float2, 1, U2, 2)    \
    X(V2, float2, 1, U4, 4)    \
    X(V4, float4, 2, U1, 1)    \
    X(V4, float4, 2, U2, 2)    \
    X(V4, float4, 2, U4, 4)

#define DEFINE_ALL_KERNELS_FOR_ONE_VU(VTag, VecT, Shift, UTag, U)           \
                                                                            \
    DEFINE_KERNEL_T(I, int,    VTag, VecT, Shift, UTag, U, C, XQ_C, YQ_C)   \
    DEFINE_KERNEL_T(I, int,    VTag, VecT, Shift, UTag, U, R, XQ_R, YQ_R)   \
    DEFINE_KERNEL_T(S, size_t, VTag, VecT, Shift, UTag, U, C, XQ_C, YQ_C)   \
    DEFINE_KERNEL_T(S, size_t, VTag, VecT, Shift, UTag, U, R, XQ_R, YQ_R)   \
                                                                            \
    DEFINE_KERNEL_M(I, int,    VTag, VecT, UTag, U, C, XQ_C, YQ_C)          \
    DEFINE_KERNEL_M(I, int,    VTag, VecT, UTag, U, R, XQ_R, YQ_R)          \
    DEFINE_KERNEL_M(S, size_t, VTag, VecT, UTag, U, C, XQ_C, YQ_C)          \
    DEFINE_KERNEL_M(S, size_t, VTag, VecT, UTag, U, R, XQ_R, YQ_R)

VU_LIST_EXCEPT_V1U1(DEFINE_ALL_KERNELS_FOR_ONE_VU)

#undef DEFINE_ALL_KERNELS_FOR_ONE_VU

// ======================================================
// Host wrappers (timeKernel + conversions no-tail for M)
// ======================================================

#define WRAP_N(IndexTag, IndexT, CRTag, nExpr)                                             \
float addNaive##IndexTag##V1U1N##CRTag(const data::DeviceData<float>& d, cudaStream_t s) { \
    return timeKernel(s, [&] {                                                             \
        addNaive##IndexTag##V1U1N##CRTag##_kernel<<<1,1,0,s>>>(nExpr, d.dx, d.dy);         \
    });                                                                                    \
}

WRAP_N(I, int,    C, static_cast<int>(d.n))
WRAP_N(I, int,    R, static_cast<int>(d.n))
WRAP_N(S, size_t, C, d.n)
WRAP_N(S, size_t, R, d.n)

#undef WRAP_N

#define WRAP_T(IndexTag, IndexT, VTag, VecT, Shift, UTag, U, CRTag, nExpr)                         \
float addNaive##IndexTag##VTag##UTag##T##CRTag(const data::DeviceData<float>& d, cudaStream_t s) { \
    return timeKernel(s, [&] {                                                                     \
        addNaive##IndexTag##VTag##UTag##T##CRTag##_kernel<<<1,1,0,s>>>(nExpr, d.dx, d.dy);         \
    });                                                                                            \
}

#define WRAP_M(IndexTag, IndexT, VTag, VecT, Shift, UTag, U, CRTag, nExpr)                         \
float addNaive##IndexTag##VTag##UTag##M##CRTag(const data::DeviceData<float>& d, cudaStream_t s) { \
    const IndexT nScalar = (IndexT)(nExpr);                                                        \
    const IndexT nVec = (IndexT)(nScalar >> (Shift));                                              \
    const VecT* xV = (const VecT*)d.dx;                                                            \
    VecT* yV = (VecT*)d.dy;                                                                        \
    return timeKernel(s, [&] {                                                                     \
        addNaive##IndexTag##VTag##UTag##M##CRTag##_kernel<<<1,1,0,s>>>(nVec, xV, yV);              \
    });                                                                                            \
}

#define WRAP_ALL_FOR_ONE_VU(VTag, VecT, Shift, UTag, U)         \
                                                                \
    WRAP_T(I, int,    VTag, VecT, Shift, UTag, U, C, (int)d.n)  \
    WRAP_T(I, int,    VTag, VecT, Shift, UTag, U, R, (int)d.n)  \
    WRAP_M(I, int,    VTag, VecT, Shift, UTag, U, C, (int)d.n)  \
    WRAP_M(I, int,    VTag, VecT, Shift, UTag, U, R, (int)d.n)  \
                                                                \
    WRAP_T(S, size_t, VTag, VecT, Shift, UTag, U, C, d.n)       \
    WRAP_T(S, size_t, VTag, VecT, Shift, UTag, U, R, d.n)       \
    WRAP_M(S, size_t, VTag, VecT, Shift, UTag, U, C, d.n)       \
    WRAP_M(S, size_t, VTag, VecT, Shift, UTag, U, R, d.n)

VU_LIST_EXCEPT_V1U1(WRAP_ALL_FOR_ONE_VU)

#undef WRAP_ALL_FOR_ONE_VU
#undef WRAP_T
#undef WRAP_M

// cleanup
#undef VU_LIST_EXCEPT_V1U1
#undef XQ_C
#undef YQ_C
#undef XQ_R
#undef YQ_R

#undef ADD2_INPLACE
#undef ADD4_INPLACE

#undef BODY_V1_U1
#undef BODY_V1_U2
#undef BODY_V1_U4
#undef BODY_V2_U1
#undef BODY_V2_U2
#undef BODY_V2_U4
#undef BODY_V4_U1
#undef BODY_V4_U2
#undef BODY_V4_U4