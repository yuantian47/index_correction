#pragma once
#ifndef RENDER_RENDER_HPP
#define RENDER_RENDER_CPP

#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include <fmt/format.h>

#include "cuda.hpp"

#include "raycast.cuh"

class CudaRenderer {
public:

    struct resolution_t {
        size_t width, height;
    };

    CudaRenderer();

    void set_volume(const int8_t* ptr, cudaExtent size, bool filter = false);

    void set_threshold(float threshold);
    void set_brightness(float brightness);

    void set_range(float tmin, float tmax);
    void set_enhancement(float edge, bool retina);

    void set_transfer_function(const std::vector<float4>& function, float offset, float scale);
    void set_transfer_function(const float4* ptr, size_t size, float offset, float scale);

    void set_cut_planes(std::vector<Plane> planes);

    void render(const Eigen::Matrix4f& modelview, const Eigen::Matrix4f& projection, const resolution_t& resolution, const region_t& region, uchar4* out);

protected:

    DeviceArray<float4> _tf_function;
    Texture _tf_tex;

    std::vector<Plane> _cut_planes;
    TransferFunction _tf;
    RenderSettings _rs;

    DeviceMemory<RenderArray> _ra_device;

    Surface _vol_surface;
    Texture _vol_tex;
    PinnedMemory<int8_t> _filter_host;
    DeviceMemory<int8_t> _filter_device_A, _filter_device_B;
    DeviceArray<int8_t> _vol_device;

    DeviceMemory<uchar4> _img_device;
};

#endif