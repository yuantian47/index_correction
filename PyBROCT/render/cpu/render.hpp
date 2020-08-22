#pragma once
#ifndef RENDER_RENDER_HPP
#define RENDER_RENDER_CPP

#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include <fmt/format.h>

#include "raycast.hpp"

class CpuRenderer {
public:

    struct resolution_t {
        size_t width, height;
    };

    CpuRenderer();

    void set_volume(const int8_t* ptr, extent_t size, bool filter);

    void set_threshold(float threshold);
    void set_brightness(float brightness);

    void set_range(float tmin, float tmax);
    void set_enhancement(float edge, bool retina);

    void set_transfer_function(const std::vector<float4>& function, float offset, float scale);
    void set_transfer_function(const float4* ptr, size_t size, float offset, float scale);

    void set_cut_planes(std::vector<Plane> planes);

    void render(const Eigen::Matrix4f& modelview, const Eigen::Matrix4f& projection, const resolution_t& resolution, const region_t& region, uchar4* out);

protected:

    std::vector<float4> _tf_function;
    Texture1D<float4> _tf_tex;

    std::vector<Plane> _cut_planes;
    TransferFunction _tf;
    RenderSettings _rs;

    NormalizedTexture3D<char> _vol_tex;
    //std::vector<char> _filter_host;
    std::vector<char> _vol_device;
};

#endif