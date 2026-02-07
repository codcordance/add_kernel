#include "thread_block.cuh"
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
// Loop selectors (mo vs gs)
// =================

// mo: just if statement
#define TB_LOOP_mo(IndexT, idxVar, base, bound, stride, BODY) \
    do { IndexT idxVar = (base); if (idxVar < (bound)) { BODY; } } while (0)

// gs: grid-stride loop
#define TB_LOOP_gs(IndexT, idxVar, base, bound, stride, BODY) \
    for (IndexT idxVar = (base); idxVar < (bound); idxVar += (stride)) { BODY; }

// =================
// Kernel generators
// =================

#define XQ_C
#define YQ_C
#define XQ_R __restrict__
#define YQ_R __restrict__

#define DEFINE_KERNEL_N(SUF, IndexTag, IndexT, CRTag, XQ, YQ) \
__global__ void addThreadBlock##IndexTag##V1U1N##CRTag##SUF##_kernel(const IndexT n, const float* XQ x, float* YQ y) { \
    const IndexT tid = (IndexT)blockIdx.x * (IndexT)blockDim.x + (IndexT)threadIdx.x; \
    const IndexT stride = (IndexT)gridDim.x * (IndexT)blockDim.x; \
    TB_LOOP_##SUF(IndexT, i, tid, n, stride, BODY_V1_U1(i, x, y)); \
}

#define DEFINE_KERNEL_T(SUF, IndexTag, IndexT, VTag, VecT, Shift, UTag, U, CRTag, XQ, YQ) \
__global__ void addThreadBlock##IndexTag##VTag##UTag##T##CRTag##SUF##_kernel(const IndexT n, const float* XQ x, float* YQ y) { \
    const IndexT tid = (IndexT)blockIdx.x * (IndexT)blockDim.x + (IndexT)threadIdx.x; \
    const IndexT totalThreads = (IndexT)gridDim.x * (IndexT)blockDim.x; \
    const IndexT strideU = totalThreads * (IndexT)(U); \
    const VecT* XQ xV = (const VecT*)x; \
    VecT* YQ yV = (VecT*)y; \
    const IndexT nVec = (IndexT)(n >> (Shift)); \
    const IndexT mainEnd = (nVec / (IndexT)(U)) * (IndexT)(U); \
    const IndexT baseU = tid * (IndexT)(U); \
     \
    TB_LOOP_##SUF(IndexT, i, baseU, mainEnd, strideU, BODY_##VTag##_##UTag(i, xV, yV)); \
     \
    TB_LOOP_##SUF(IndexT, j, mainEnd + tid, nVec, totalThreads, BODY_##VTag##_U1(j, xV, yV)); \
     \
    const IndexT tailStart = (IndexT)(nVec << (Shift)); \
    TB_LOOP_##SUF(IndexT, k, tailStart + tid, n, totalThreads, BODY_V1_U1(k, x, y)); \
}

#define DEFINE_KERNEL_M(SUF, IndexTag, IndexT, VTag, VecT, UTag, U, CRTag, XQ, YQ) \
__global__ void addThreadBlock##IndexTag##VTag##UTag##M##CRTag##SUF##_kernel(const IndexT nVec, const VecT* XQ xV, VecT* YQ yV) { \
    const IndexT tid = (IndexT)blockIdx.x * (IndexT)blockDim.x + (IndexT)threadIdx.x; \
    const IndexT totalThreads = (IndexT)gridDim.x * (IndexT)blockDim.x; \
    const IndexT strideU = totalThreads * (IndexT)(U); \
    const IndexT baseU = tid * (IndexT)(U); \
    TB_LOOP_##SUF(IndexT, i, baseU, nVec, strideU, BODY_##VTag##_##UTag(i, xV, yV)); \
}

// ===========================
// Kernel definition w/ tables
// ===========================

// N: I/S x C/R x (mo/gs)
#define DEFINE_N_BOTH(IndexTag, IndexT, CRTag, XQ, YQ) \
    DEFINE_KERNEL_N(mo, IndexTag, IndexT, CRTag, XQ, YQ) \
    DEFINE_KERNEL_N(gs, IndexTag, IndexT, CRTag, XQ, YQ)

DEFINE_N_BOTH(I, int,    C, XQ_C, YQ_C)
DEFINE_N_BOTH(I, int,    R, XQ_R, YQ_R)
DEFINE_N_BOTH(S, size_t, C, XQ_C, YQ_C)
DEFINE_N_BOTH(S, size_t, R, XQ_R, YQ_R)

#undef DEFINE_N_BOTH

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

#define DEFINE_ALL_KERNELS_FOR_ONE_VU(VTag, VecT, Shift, UTag, U)                 \
                                                                                  \
    DEFINE_KERNEL_T(mo, I, int,    VTag, VecT, Shift, UTag, U, C, XQ_C, YQ_C)     \
    DEFINE_KERNEL_T(gs, I, int,    VTag, VecT, Shift, UTag, U, C, XQ_C, YQ_C)     \
    DEFINE_KERNEL_T(mo, I, int,    VTag, VecT, Shift, UTag, U, R, XQ_R, YQ_R)     \
    DEFINE_KERNEL_T(gs, I, int,    VTag, VecT, Shift, UTag, U, R, XQ_R, YQ_R)     \
    DEFINE_KERNEL_T(mo, S, size_t, VTag, VecT, Shift, UTag, U, C, XQ_C, YQ_C)     \
    DEFINE_KERNEL_T(gs, S, size_t, VTag, VecT, Shift, UTag, U, C, XQ_C, YQ_C)     \
    DEFINE_KERNEL_T(mo, S, size_t, VTag, VecT, Shift, UTag, U, R, XQ_R, YQ_R)     \
    DEFINE_KERNEL_T(gs, S, size_t, VTag, VecT, Shift, UTag, U, R, XQ_R, YQ_R)     \
                                                                                  \
    DEFINE_KERNEL_M(mo, I, int,    VTag, VecT, UTag, U, C, XQ_C, YQ_C)            \
    DEFINE_KERNEL_M(gs, I, int,    VTag, VecT, UTag, U, C, XQ_C, YQ_C)            \
    DEFINE_KERNEL_M(mo, I, int,    VTag, VecT, UTag, U, R, XQ_R, YQ_R)            \
    DEFINE_KERNEL_M(gs, I, int,    VTag, VecT, UTag, U, R, XQ_R, YQ_R)            \
    DEFINE_KERNEL_M(mo, S, size_t, VTag, VecT, UTag, U, C, XQ_C, YQ_C)            \
    DEFINE_KERNEL_M(gs, S, size_t, VTag, VecT, UTag, U, C, XQ_C, YQ_C)            \
    DEFINE_KERNEL_M(mo, S, size_t, VTag, VecT, UTag, U, R, XQ_R, YQ_R)            \
    DEFINE_KERNEL_M(gs, S, size_t, VTag, VecT, UTag, U, R, XQ_R, YQ_R)

VU_LIST_EXCEPT_V1U1(DEFINE_ALL_KERNELS_FOR_ONE_VU)

#undef DEFINE_ALL_KERNELS_FOR_ONE_VU

// =====================
// Host wrappers (launch)
// =====================

static int divUpSizeT(const std::size_t a, const std::size_t b) {
    return (int)((a + b - 1) / b);
}

static int blocksFor(const std::size_t nVec, const int u, const int blockSize) {
    std::size_t threadsNeeded = (nVec + (std::size_t)u - 1) / (std::size_t)u;
    if (threadsNeeded == 0) threadsNeeded = 1;
    int blocks = divUpSizeT(threadsNeeded, (std::size_t)blockSize);
    if (blocks < 1) blocks = 1;
    return blocks;
}

#define WRAP_N_MO(IndexTag, IndexT, CRTag, nExpr) \
float addThreadBlock##IndexTag##V1U1N##CRTag##mo(const data::DeviceData<float>& d, cudaStream_t s, const int blockSize) { \
    const std::size_t nScalar = (std::size_t)(nExpr); \
    const int numBlocks = divUpSizeT(nScalar, (std::size_t)blockSize); \
    return timeKernel(s, [&] { \
        addThreadBlock##IndexTag##V1U1N##CRTag##mo##_kernel<<<numBlocks, blockSize, 0, s>>>((IndexT)(nExpr), d.dx, d.dy); \
    }); \
}

#define WRAP_N_GS(IndexTag, IndexT, CRTag, nExpr) \
float addThreadBlock##IndexTag##V1U1N##CRTag##gs(const data::DeviceData<float>& d, cudaStream_t s, const int blockSize, const int numBlocks) { \
    return timeKernel(s, [&] { \
        addThreadBlock##IndexTag##V1U1N##CRTag##gs##_kernel<<<numBlocks, blockSize, 0, s>>>((IndexT)(nExpr), d.dx, d.dy); \
    }); \
}

WRAP_N_MO(I, int,    C, static_cast<int>(d.n))
WRAP_N_MO(I, int,    R, static_cast<int>(d.n))
WRAP_N_MO(S, size_t, C, d.n)
WRAP_N_MO(S, size_t, R, d.n)

WRAP_N_GS(I, int,    C, static_cast<int>(d.n))
WRAP_N_GS(I, int,    R, static_cast<int>(d.n))
WRAP_N_GS(S, size_t, C, d.n)
WRAP_N_GS(S, size_t, R, d.n)

#undef WRAP_N_MO
#undef WRAP_N_GS

#define WRAP_T_MO(IndexTag, IndexT, VTag, VecT, Shift, UTag, U, CRTag, nExpr) \
float addThreadBlock##IndexTag##VTag##UTag##T##CRTag##mo(const data::DeviceData<float>& d, cudaStream_t s, const int blockSize) { \
    const std::size_t nScalar = (std::size_t)(nExpr); \
    const std::size_t nVec = nScalar >> (Shift); \
    const int numBlocks = blocksFor(nVec, (U), blockSize); \
    return timeKernel(s, [&] { \
        addThreadBlock##IndexTag##VTag##UTag##T##CRTag##mo##_kernel<<<numBlocks, blockSize, 0, s>>>((IndexT)(nExpr), d.dx, d.dy); \
    }); \
}

#define WRAP_T_GS(IndexTag, IndexT, VTag, VecT, Shift, UTag, U, CRTag, nExpr) \
float addThreadBlock##IndexTag##VTag##UTag##T##CRTag##gs(const data::DeviceData<float>& d, cudaStream_t s, const int blockSize, const int numBlocks) { \
    return timeKernel(s, [&] { \
        addThreadBlock##IndexTag##VTag##UTag##T##CRTag##gs##_kernel<<<numBlocks, blockSize, 0, s>>>((IndexT)(nExpr), d.dx, d.dy); \
    }); \
}

#define WRAP_M_MO(IndexTag, IndexT, VTag, VecT, Shift, UTag, U, CRTag, nExpr) \
float addThreadBlock##IndexTag##VTag##UTag##M##CRTag##mo(const data::DeviceData<float>& d, cudaStream_t s, const int blockSize) { \
    const std::size_t nScalar = (std::size_t)(nExpr); \
    const std::size_t nVecSZ = nScalar >> (Shift); \
    const int numBlocks = blocksFor(nVecSZ, (U), blockSize); \
    const IndexT nVec = (IndexT)nVecSZ; \
    const VecT* xV = (const VecT*)d.dx; \
    VecT* yV = (VecT*)d.dy; \
    return timeKernel(s, [&] { \
        addThreadBlock##IndexTag##VTag##UTag##M##CRTag##mo##_kernel<<<numBlocks, blockSize, 0, s>>>(nVec, xV, yV); \
    }); \
}

#define WRAP_M_GS(IndexTag, IndexT, VTag, VecT, Shift, UTag, U, CRTag, nExpr) \
float addThreadBlock##IndexTag##VTag##UTag##M##CRTag##gs(const data::DeviceData<float>& d, cudaStream_t s, const int blockSize, const int numBlocks) { \
    const std::size_t nScalar = (std::size_t)(nExpr); \
    const IndexT nVec = (IndexT)(nScalar >> (Shift)); \
    const VecT* xV = (const VecT*)d.dx; \
    VecT* yV = (VecT*)d.dy; \
    return timeKernel(s, [&] { \
        addThreadBlock##IndexTag##VTag##UTag##M##CRTag##gs##_kernel<<<numBlocks, blockSize, 0, s>>>(nVec, xV, yV); \
    }); \
}

#define WRAP_ALL_FOR_ONE_VU(VTag, VecT, Shift, UTag, U)           \
                                                                  \
    WRAP_T_MO(I, int,    VTag, VecT, Shift, UTag, U, C, (int)d.n) \
    WRAP_T_MO(I, int,    VTag, VecT, Shift, UTag, U, R, (int)d.n) \
    WRAP_T_GS(I, int,    VTag, VecT, Shift, UTag, U, C, (int)d.n) \
    WRAP_T_GS(I, int,    VTag, VecT, Shift, UTag, U, R, (int)d.n) \
    WRAP_M_MO(I, int,    VTag, VecT, Shift, UTag, U, C, (int)d.n) \
    WRAP_M_MO(I, int,    VTag, VecT, Shift, UTag, U, R, (int)d.n) \
    WRAP_M_GS(I, int,    VTag, VecT, Shift, UTag, U, C, (int)d.n) \
    WRAP_M_GS(I, int,    VTag, VecT, Shift, UTag, U, R, (int)d.n) \
                                                                  \
    WRAP_T_MO(S, size_t, VTag, VecT, Shift, UTag, U, C, d.n)      \
    WRAP_T_MO(S, size_t, VTag, VecT, Shift, UTag, U, R, d.n)      \
    WRAP_T_GS(S, size_t, VTag, VecT, Shift, UTag, U, C, d.n)      \
    WRAP_T_GS(S, size_t, VTag, VecT, Shift, UTag, U, R, d.n)      \
    WRAP_M_MO(S, size_t, VTag, VecT, Shift, UTag, U, C, d.n)      \
    WRAP_M_MO(S, size_t, VTag, VecT, Shift, UTag, U, R, d.n)      \
    WRAP_M_GS(S, size_t, VTag, VecT, Shift, UTag, U, C, d.n)      \
    WRAP_M_GS(S, size_t, VTag, VecT, Shift, UTag, U, R, d.n)

VU_LIST_EXCEPT_V1U1(WRAP_ALL_FOR_ONE_VU)

#undef WRAP_ALL_FOR_ONE_VU
#undef WRAP_T_MO
#undef WRAP_T_GS
#undef WRAP_M_MO
#undef WRAP_M_GS

// cleanup
#undef VU_LIST_EXCEPT_V1U1
#undef DEFINE_KERNEL_N
#undef DEFINE_KERNEL_T
#undef DEFINE_KERNEL_M

#undef TB_LOOP_mo
#undef TB_LOOP_gs

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
