#pragma once
#ifndef RAYCAST_OPTIMIZED_CUH
#define RAYCAST_OPTIMIZED_CUH

#include <cstdint>
#include <cuda_runtime_api.h>

struct TransferFunction {
    cudaTextureObject_t tex;
    float threshold, offset, scale;
};

struct RenderSettings {
    float edge_enhancement;
    float brightness;
    bool retina;
    float tmin, tmax;
    float tstep;
    int max_steps;
    float opacity_threshold;
    float surface_threshold;
};

struct Plane {
    float3 point, normal;
};

#define PlaneArray_MAX      32ULL

struct PlaneArray {
    unsigned char count;
    Plane planes[PlaneArray_MAX];
};

#define RenderArray_MAX         16ULL

struct float4x4 {
    float4 m[4];
};

template<typename T>
struct range_t {
    T min, max;

    __device__ __host__
    T range() const {
        return max - min;
    }
};

struct region_t {
    range_t<float> u;
    range_t<float> v;
};

struct Render {
    uchar4* color;
    float* depth;
    size_t width, height;
    PlaneArray pa;
    float4x4 inverse_modelview, inverse_projection;
    region_t region;
};

struct RenderArray {
    unsigned char count;
    Render renders[RenderArray_MAX];
};

struct Ray {
    float3 origin, normal;
};

struct CastResult {
    float depth;
    float4 color;
};

__device__
CastResult raycast(cudaTextureObject_t volume, const Ray& ray, float tnear, float tfar, const TransferFunction& tf, const RenderSettings& rs);

__device__
bool check_planes(const Plane* planes, unsigned char count, const Ray& ray, float* tnear, float* tfar, unsigned char* pnear, unsigned char* pfar);

__device__
void render(uchar4* color, float* depth, unsigned int width, unsigned int height, const PlaneArray& planes, cudaTextureObject_t volume, const float4x4& inverse_modelview, const float4x4& inverse_projection, const region_t& region, const TransferFunction& tf, const RenderSettings& rs);

__global__
void multirender(RenderArray* ra, cudaTextureObject_t volume, TransferFunction tf, RenderSettings rs);

__global__
void multirender(RenderArray* ra, unsigned int index, cudaTextureObject_t volume, TransferFunction tf, RenderSettings rs);

void launch_multirender(const dim3& blocks, const dim3& threads, RenderArray* ra, cudaTextureObject_t volume, const TransferFunction& tf, const RenderSettings& rs);

#endif
