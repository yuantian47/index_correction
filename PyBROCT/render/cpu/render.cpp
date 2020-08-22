#include "render.hpp"

#include "filter.hpp"

CpuRenderer::CpuRenderer() {
    // initialize render settings
    _rs.edge_enhancement = 1;
    _rs.brightness = 1;
    _rs.retina = true;
    _rs.tmin = 0;
    _rs.tmax = 10;
    _rs.tstep = 0.005f;
    _rs.max_steps = 2000;
    _rs.opacity_threshold = 0.95f;
    _rs.surface_threshold = 0.1f;

    // initialize transfer function
    _tf.offset = 0;
    _tf.scale = 1;
    _tf.threshold = 0.3f;

    std::vector<float4> function = {
        make_float4(0),
        make_float4(0.0125f),
        make_float4(0.025f),
        make_float4(0.05f),
        make_float4(0.10f),
        make_float4(0.15f),
        make_float4(0.25f),
        make_float4(0.35f),
        make_float4(0.55f),
        make_float4(0.75f),
        make_float4(0.85f),
        make_float4(0.95f),
        make_float4(0.98f),
        make_float4(1.0f),
        make_float4(1.0f),
        make_float4(1.0f),
        make_float4(1.0f),
        make_float4(1.0f),
    };
    set_transfer_function(function, 0, 1);

    std::vector<Plane> cut_planes = {
        { { 0, 0, 1 },  { 0, 0, -1 } },
        { { 0, 0, -1 }, { 0, 0, 1 } },
        { { 0, 1, 0 },  { 0, -1, 0 } },
        { { 0, -1, 0 }, { 0, 1, 0 } },
        { { 1, 0, 0 },  { -1, 0, 0 } },
        { { -1, 0, 0 }, { 1, 0, 0 } },
    };
    set_cut_planes(std::move(cut_planes));
}

bool operator==(extent_t a, extent_t b) {
    return a.width == b.width && a.height == b.height && a.depth == b.depth;
}
bool operator!=(extent_t a, extent_t b) {
    return !(a == b);
}

void CpuRenderer::set_volume(const int8_t* ptr, extent_t size, bool filter) {
    // copy over for filtering kernel

    size_t SperA = size.height;
    size_t AperB = size.depth;
    size_t BperV = size.width;

    extent_t extent = { size.depth, size.height, size.width };

    // prepare volume buffer
    if(_vol_tex.extent != extent) {
        _vol_device.resize(extent.count());
        _vol_tex = { _vol_device.data(), extent };
    }

    if(filter) {
        //if(_filter_host.size() != extent.count()) {
        //    _filter_host.resize(extent.count());
        //}

        // median filtering in 3D
        median_filter(3, _vol_device.data(), (char*)ptr, extent);

        //// Gaussian filtering of B-scans
        //{
        //    dim3 threads;
        //    dim3 blocks;
        //    threads.x = COLS_X;
        //    threads.y = ROWS_Y;
        //    threads.z = 1;
        //    blocks.x = static_cast<unsigned int>(std::ceil(float(AperB) / (COLS_X*PIX_PER_THREAD)));
        //    blocks.y = static_cast<unsigned int>(std::ceil(float(SperA) / ROWS_Y));
        //    blocks.z = 1;

        //    for(size_t i = 0; i < BperV; i++) {
        //        size_t offset = AperB * SperA * i;
        //        launch_convolutionRows(blocks, threads, (char*)_filter_device_A.ptr() + offset, (char*)_filter_device_B.ptr() + offset, AperB, SperA, AperB);
        //        cudaError_t error = cudaGetLastError();
        //        if(error != cudaSuccess) {
        //            throw std::runtime_error(fmt::format("unable to launch kernel: {}", cudaGetErrorName(error)));
        //        }
        //    }
        //}
        //
        //{
        //    dim3 threads;
        //    dim3 blocks;
        //    threads.x = ROWS_X;
        //    threads.y = COLS_Y;
        //    threads.z = 1;
        //    blocks.x = static_cast<unsigned int>(std::ceil(float(AperB) / ROWS_X));
        //    blocks.y = static_cast<unsigned int>(std::ceil(float(SperA) / (COLS_Y*PIX_PER_THREAD)));
        //    blocks.z = 1;

        //    for(size_t i = 0; i < BperV; i++) {
        //        size_t offset = AperB * SperA * i;
        //        launch_convolutionCols(blocks, threads, _vol_surface.handle(), (char*)_filter_device_A.ptr() + offset, AperB, SperA, AperB, i);
        //        cudaError_t error = cudaGetLastError();
        //        if(error != cudaSuccess) {
        //            throw std::runtime_error(fmt::format("unable to launch kernel: {}", cudaGetErrorName(error)));
        //        }
        //    }
        //}
    } else {
        // copy over directly
        std::memcpy(_vol_device.data(), ptr, _vol_device.size() * sizeof(decltype(_vol_device)::value_type));
    }
}

void CpuRenderer::set_threshold(float threshold) {
    _tf.threshold = threshold;
}

void CpuRenderer::set_brightness(float brightness) {
    _rs.brightness = brightness;
}

void CpuRenderer::set_range(float tmin, float tmax) {
    _rs.tmin = tmin;
    _rs.tmax = tmax;
}

void CpuRenderer::set_enhancement(float edge, bool retina) {
    _rs.edge_enhancement = edge;
    _rs.retina = retina;
}

void CpuRenderer::set_transfer_function(const std::vector<float4>& function, float offset, float scale) {
    set_transfer_function(function.data(), function.size(), offset, scale);
}

void CpuRenderer::set_transfer_function(const float4* ptr, size_t size, float offset, float scale) {
    _tf.offset = offset;
    _tf.scale = scale;

    // copy over
    if(_tf_tex.length != size) {
        _tf_function.resize(size);
    }
    std::memcpy(_tf_function.data(), ptr, _tf_function.size() * sizeof(decltype(_tf_function)::value_type));

    // bind texture
    _tf_tex = { _tf_function.data(), size };
    _tf.function = &_tf_tex;
}

void CpuRenderer::set_cut_planes(std::vector<Plane> cut_planes) {
    if(cut_planes.size() > PlaneArray_MAX) {
        throw std::runtime_error(fmt::format("too many cut planes: {} > {}", cut_planes.size(), PlaneArray_MAX));
    }

    _cut_planes = std::move(cut_planes);
}

void CpuRenderer::render(const Eigen::Matrix4f& modelview, const Eigen::Matrix4f& projection, const resolution_t& resolution, const region_t& region, uchar4* out) {
    std::memset(out, 0, resolution.width * resolution.height * sizeof(uchar4));

    if(_vol_device.empty()) {
        return;
    }

    RenderArray ra;
    ra.count = 1;
    auto& render = ra.renders[0];

    render.color = out;
    render.depth = nullptr;
    render.width = unsigned int(resolution.width);
    render.height = unsigned int(resolution.height);

    render.pa.count = unsigned char(_cut_planes.size());
    std::copy(_cut_planes.begin(), _cut_planes.end(), render.pa.planes);

    Eigen::Matrix<float, 4, 4, Eigen::RowMajor>::Map(&render.inverse_modelview.m[0].x) = modelview.inverse();
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor>::Map(&render.inverse_projection.m[0].x) = projection.inverse();

    render.region = region;

    // launch the render
    multirender(&ra, _vol_tex, _tf, _rs);
}