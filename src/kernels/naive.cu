#include "naive.cuh"
#include "../cuda_utils.h"

__global__ void addNaiveKernel(const int n, const float* x, float* y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

__global__ void addNaiveRestrictKernel(const int n, const float* __restrict__ x, float* __restrict__ y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

__global__ void addNaiveSizeTKernel(const size_t n, const float* x, float* y) {
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

__global__ void addNaiveSizeTRestrictKernel(const size_t n, const float* __restrict__ x, float* __restrict__ y) {
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

__global__ void addNaiveFloat2RestrictKernel(const int n, const float* __restrict__ x, float* __restrict__ y) {
    const int n2 = n >> 1;

    const float2* __restrict__ x2 = reinterpret_cast<const float2*>(x);
    float2* __restrict__ y2 = reinterpret_cast<float2*>(y);

    for (int i = 0; i < n2; ++i) {
        float2 a = x2[i];
        float2 b = y2[i];
        b.x += a.x;
        b.y += a.y;
        y2[i] = b;
    }

    // tail
    if (n & 1) {
        const int i = n - 1;
        y[i] = x[i] + y[i];
    }
}

__global__ void addNaiveFloat2RestrictNoTailKernel(const int n2, const float2* __restrict__ x2, float2* __restrict__ y2) {
    for (int i = 0; i < n2; ++i) {
        float2 a = x2[i];
        float2 b = y2[i];
        b.x += a.x;
        b.y += a.y;
        y2[i] = b;
    }
}

__global__ void addNaiveFloat4RestrictKernel(const int n, const float* __restrict__ x, float* __restrict__ y) {
    const int n4 = n >> 2;

    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    float4* __restrict__ y4 = reinterpret_cast<float4*>(y);

    for (int i = 0; i < n4; ++i) {
        float4 a = x4[i];
        float4 b = y4[i];
        b.x += a.x;
        b.y += a.y;
        b.z += a.z;
        b.w += a.w;
        y4[i] = b;
    }

    for (int i = (n4 << 2); i < n; ++i) {
        y[i] = x[i] + y[i];
    }
}

__global__ void addNaiveFloat4NoTailKernel(const int n4, const float4* x4, float4* y4) {
    for (int i = 0; i < n4; ++i) {
        float4 a = x4[i];
        float4 b = y4[i];
        b.x += a.x;
        b.y += a.y;
        b.z += a.z;
        b.w += a.w;
        y4[i] = b;
    }
}

__global__ void addNaiveFloat4RestrictNoTailKernel(const int n4, const float4* __restrict__ x4, float4* __restrict__ y4) {
    for (int i = 0; i < n4; ++i) {
        float4 a = x4[i];
        float4 b = y4[i];
        b.x += a.x;
        b.y += a.y;
        b.z += a.z;
        b.w += a.w;
        y4[i] = b;
    }
}


float addNaive(const data::DeviceData<float> &d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveKernel<<<1, 1, 0, s>>>(static_cast<int>(d.n), d.dx, d.dy);
    });
}

float addNaiveRestrict(const data::DeviceData<float> &d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveRestrictKernel<<<1, 1, 0, s>>>(static_cast<int>(d.n), d.dx, d.dy);
    });
}

float addNaiveSizeT(const data::DeviceData<float> &d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveSizeTKernel<<<1, 1, 0, s>>>(d.n, d.dx, d.dy);
    });
}

float addNaiveSizeTRestrict(const data::DeviceData<float> &d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveSizeTRestrictKernel<<<1, 1, 0, s>>>(d.n, d.dx, d.dy);
    });
}

float addNaiveFloat2Restrict(const data::DeviceData<float>& d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveFloat2RestrictKernel<<<1, 1, 0, s>>>(
            static_cast<int>(d.n), d.dx, d.dy
        );
    });
}

float addNaiveFloat2RestrictNoTail(const data::DeviceData<float>& d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveFloat2RestrictNoTailKernel<<<1, 1, 0, s>>>(
            static_cast<int>(d.n) >> 1, reinterpret_cast<const float2*>(d.dx), reinterpret_cast<float2*>(d.dy)
        );
    });
}

float addNaiveFloat4Restrict(const data::DeviceData<float>& d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveFloat4RestrictKernel<<<1, 1, 0, s>>>(
            static_cast<int>(d.n), d.dx, d.dy
        );
    });
}

float addNaiveFloat4NoTail(const data::DeviceData<float>& d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveFloat4RestrictNoTailKernel<<<1, 1, 0, s>>>(
            static_cast<int>(d.n) >> 2, reinterpret_cast<const float4*>(d.dx), reinterpret_cast<float4*>(d.dy)
        );
    });
}

float addNaiveFloat4RestrictNoTail(const data::DeviceData<float>& d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveFloat4RestrictNoTailKernel<<<1, 1, 0, s>>>(
            static_cast<int>(d.n) >> 2, reinterpret_cast<const float4*>(d.dx), reinterpret_cast<float4*>(d.dy)
        );
    });
}

