#pragma once

#include "../data.h"

float addNaive(const data::DeviceData<float> &d, cudaStream_t s);
float addNaiveRestrict(const data::DeviceData<float> &d, cudaStream_t s);
float addNaiveUnroll4NoTailRestrict(const data::DeviceData<float> &d, cudaStream_t s);
float addNaiveSizeT(const data::DeviceData<float> &d, cudaStream_t s);
float addNaiveSizeTRestrict(const data::DeviceData<float> &d, cudaStream_t s);
float addNaiveUnroll2Restrict(const data::DeviceData<float>& d, cudaStream_t s);
float addNaiveUnroll2MultRestrict(const data::DeviceData<float>& d, cudaStream_t s);
float addNaiveUnroll4Restrict(const data::DeviceData<float>& d, cudaStream_t s);
float addNaiveUnroll4Mult(const data::DeviceData<float>& d, cudaStream_t s);
float addNaiveFloat4NoTailRestrict(const data::DeviceData<float>& d, cudaStream_t s);

