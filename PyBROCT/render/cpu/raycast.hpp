#pragma once
#ifndef RAYCAST_OPTIMIZED_CUH
#define RAYCAST_OPTIMIZED_CUH

#include <cstdint>
#include <vector>

#include <Eigen/Core>

struct extent_t {
    size_t width, height, depth;

    size_t count() {
        return width * height * depth;
    }
};

struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
struct uchar4 { unsigned char x, y, w, z; };

inline float4 make_float4(float o) {
    return { o, o, o, o };
}

inline float4 operator+(const float4& a, const float4& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

inline float3 operator*(const float3& a, const float3& b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}


inline float3 operator/(const float3& a, const float3& b) {
    return { a.x / b.x, a.y / b.y, a.z / b.z };
}

template<typename T>
T clamp(const T& o) {
    if(o < 0) {
        return 0;
    } else if(o > 1) {
        return 1;
    } else {
        return o;
    }
}

template<typename T>
struct Texture1D {
    T* data;
    size_t length;

    T at(size_t xi) const {
        xi = std::min(xi, length - 1);
        return data[xi];
    }

    T at(float x) const {
        float xi = clamp(x) * (length - 1);
        auto idx = size_t(xi);
        float r = xi - idx;

        return (1 - r) * at(idx) + r * at(idx + 1);
    }
};

template<typename T>
struct NormalizedTexture3D {
    T* data;
    extent_t extent;

    float at(size_t xi, size_t yi, size_t zi) const {
        xi = std::min(xi, extent.width - 1);
        yi = std::min(yi, extent.height - 1);
        zi = std::min(zi, extent.depth - 1);
        return data[zi * extent.height * extent.width + yi * extent.width + xi];
    }

    float at(float x, float y, float z) const {
        float xi = clamp(x) * (extent.width - 1);
        auto idx = size_t(xi);
        float rx = xi - idx;
        float yi = clamp(y) * (extent.height - 1);
        auto idy = size_t(yi);
        float ry = yi - idy;
        float zi = clamp(z) * (extent.depth - 1);
        auto idz = size_t(zi);
        float rz = zi - idz;

        auto c00 = at(idx, idy, idz) * (1 - rx) + at(idx + 1, idy, idz) * rx;
        auto c01 = at(idx, idy, idz + 1) * (1 - rx) + at(idx + 1, idy, idz + 1) * rx;
        auto c10 = at(idx, idy + 1, idz) * (1 - rx) + at(idx + 1, idy + 1, idz) * rx;
        auto c11 = at(idx, idy + 1, idz + 1) * (1 - rx) + at(idx + 1, idy + 1, idz + 1) * rx;

        auto c0 = c00 * (1 - ry) + c10 * ry;
        auto c1 = c01 * (1 - ry) + c11 * ry;

        auto c = c0 * (1 - rz) + c1 * rz;

        return c / float(std::numeric_limits<T>::max());
    }
};

struct TransferFunction {
    Texture1D<float4>* function;
    float threshold, offset, scale;
};

template<typename T>
struct range_t {
    T min, max;

    T range() const {
        return max - min;
    }
};

struct region_t {
    range_t<float> u;
    range_t<float> v;
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

struct Render {
    uchar4* color;
    float* depth;
    unsigned int width, height;
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

CastResult raycast(const NormalizedTexture3D<char>& volume, const Ray& ray, float tnear, float tfar, const TransferFunction& tf, const RenderSettings& rs);

bool check_planes(const Plane* planes, unsigned char count, const Ray& ray, float* tnear, float* tfar, unsigned char* pnear, unsigned char* pfar);

void render(uchar4* color, float* depth, unsigned int width, unsigned int height, const PlaneArray& planes, const NormalizedTexture3D<char>& volume, const float4x4& inverse_modelview, const float4x4& inverse_projection, const region_t& region, const TransferFunction& tf, const RenderSettings& rs);

void multirender(RenderArray* ra, const NormalizedTexture3D<char>& volume, TransferFunction tf, RenderSettings rs);

#endif
