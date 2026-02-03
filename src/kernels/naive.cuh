#pragma once

#include "../data.h"

float addNaive(const data::DeviceData<float> &d, cudaStream_t s);
float addNaiveRestrict(const data::DeviceData<float> &d, cudaStream_t s);
float addNaiveSizeT(const data::DeviceData<float> &d, cudaStream_t s);
float addNaiveSizeTRestrict(const data::DeviceData<float> &d, cudaStream_t s);
float addNaiveFloat2Restrict(const data::DeviceData<float>& d, cudaStream_t s);
float addNaiveFloat2RestrictNoTail(const data::DeviceData<float>& d, cudaStream_t s);
float addNaiveFloat4Restrict(const data::DeviceData<float>& d, cudaStream_t s);
float addNaiveFloat4NoTail(const data::DeviceData<float>& d, cudaStream_t s);
float addNaiveFloat4RestrictNoTail(const data::DeviceData<float>& d, cudaStream_t s);

