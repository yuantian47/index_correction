#include "render.hpp"

#include "filter.cuh"
#include "helper_math.hpp"

CudaRenderer::CudaRenderer() {
    _ra_device.allocate(1);

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

bool operator==(cudaExtent a, cudaExtent b) {
    return a.width == b.width && a.height == b.height && a.depth == b.depth;
}
bool operator!=(cudaExtent a, cudaExtent b) {
    return !(a == b);
}

void CudaRenderer::set_volume(const int8_t* ptr, cudaExtent size, bool filter) {
    // copy over for filtering kernel
    size_t count = size.width * size.height * size.depth;

    size_t SperA = size.height;
    size_t AperB = size.depth;
    size_t BperV = size.width;

    cudaExtent extent = { size.depth, size.height, size.width };

    // prepare volume buffer
    if(_vol_device.extent() != extent) {
        _vol_tex.reset();
        _vol_device.allocate(extent);

        // bind surface
        _vol_surface = _vol_device.surface();

        // bind texture
        cudaTextureDesc desc;
        std::memset(&desc, 0, sizeof(desc));

        desc.addressMode[0] = cudaAddressModeClamp;
        desc.addressMode[1] = cudaAddressModeClamp;
        desc.addressMode[2] = cudaAddressModeClamp;
        desc.filterMode = cudaFilterModeLinear;
        desc.readMode = cudaReadModeNormalizedFloat;
        desc.normalizedCoords = 1;

        _vol_tex = _vol_device.texture(desc);
    }

    if(filter) {
        if(_filter_device_A.count() != count) {
            _filter_device_A.allocate(count);
            _filter_device_B.allocate(count);
            _filter_host.allocate(count);
        }
        _filter_device_A.write(ptr);

        // median filtering in 3D
        {
            dim3 threads = { 16, 32, 1 };
            dim3 blocks;
            blocks.x = static_cast<unsigned int>(std::ceil(AperB / (float)threads.x));
            blocks.y = static_cast<unsigned int>(std::ceil(SperA / (float)threads.y));
            blocks.z = 1;

            launch_medKernel(blocks, threads, (char*)_filter_device_B.ptr(), (char*)_filter_device_A.ptr(), SperA, AperB, BperV, AperB);
            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess) {
                throw std::runtime_error(fmt::format("unable to launch kernel: {}", cudaGetErrorName(error)));
            }
        }

        // Gaussian filtering of B-scans
        {
            dim3 threads;
            dim3 blocks;
            threads.x = COLS_X;
            threads.y = ROWS_Y;
            threads.z = 1;
            blocks.x = static_cast<unsigned int>(std::ceil(float(AperB) / (COLS_X*PIX_PER_THREAD)));
            blocks.y = static_cast<unsigned int>(std::ceil(float(SperA) / ROWS_Y));
            blocks.z = 1;

            for(size_t i = 0; i < BperV; i++) {
                size_t offset = AperB * SperA * i;
                launch_convolutionRows(blocks, threads, (char*)_filter_device_A.ptr() + offset, (char*)_filter_device_B.ptr() + offset, AperB, SperA, AperB);
                cudaError_t error = cudaGetLastError();
                if(error != cudaSuccess) {
                    throw std::runtime_error(fmt::format("unable to launch kernel: {}", cudaGetErrorName(error)));
                }
            }
        }

        {
            dim3 threads;
            dim3 blocks;
            threads.x = ROWS_X;
            threads.y = COLS_Y;
            threads.z = 1;
            blocks.x = static_cast<unsigned int>(std::ceil(float(AperB) / ROWS_X));
            blocks.y = static_cast<unsigned int>(std::ceil(float(SperA) / (COLS_Y*PIX_PER_THREAD)));
            blocks.z = 1;

            for(size_t i = 0; i < BperV; i++) {
                size_t offset = AperB * SperA * i;
                launch_convolutionCols(blocks, threads, _vol_surface.handle(), (char*)_filter_device_A.ptr() + offset, AperB, SperA, AperB, i);
                cudaError_t error = cudaGetLastError();
                if(error != cudaSuccess) {
                    throw std::runtime_error(fmt::format("unable to launch kernel: {}", cudaGetErrorName(error)));
                }
            }
        }
    } else {
        // copy over directly
        _vol_device.write(ptr);
    }
}

void CudaRenderer::set_threshold(float threshold) {
    _tf.threshold = threshold;
}

void CudaRenderer::set_brightness(float brightness) {
    _rs.brightness = brightness;
}

void CudaRenderer::set_range(float tmin, float tmax) {
    _rs.tmin = tmin;
    _rs.tmax = tmax;
}

void CudaRenderer::set_enhancement(float edge, bool retina) {
    _rs.edge_enhancement = edge;
    _rs.retina = retina;
}

void CudaRenderer::set_transfer_function(const std::vector<float4>& function, float offset, float scale) {
    set_transfer_function(function.data(), function.size(), offset, scale);
}

void CudaRenderer::set_transfer_function(const float4* ptr, size_t size, float offset, float scale) {
    _tf.offset = offset;
    _tf.scale = scale;

    // copy over
    if(_tf_function.count() != size) {
        _tf_tex.reset();
        _tf_function.allocate(size);
    }
    _tf_function.write(ptr);

    // bind texture
    _tf_tex = _tf_function.texture(cudaAddressModeClamp, cudaFilterModeLinear, cudaReadModeElementType);
    _tf.tex = _tf_tex.handle();
}

void CudaRenderer::set_cut_planes(std::vector<Plane> cut_planes) {
    if(cut_planes.size() > PlaneArray_MAX) {
        throw std::runtime_error(fmt::format("too many cut planes: {} > {}", cut_planes.size(), PlaneArray_MAX));
    }

    _cut_planes = std::move(cut_planes);
}

void CudaRenderer::render(const Eigen::Matrix4f& modelview, const Eigen::Matrix4f& projection, const resolution_t& resolution, const region_t& region, uchar4* out) {
    // allocate image output
    if(_img_device.count() != resolution.width * resolution.height) {
        _img_device.allocate(resolution.width * resolution.height);
    }
    cudaMemset(_img_device.ptr(), 0, _img_device.size());

    if(!_vol_tex.valid()) {
        return;
    }

    RenderArray ra;
    ra.count = 1;
    auto& render = ra.renders[0];

    render.color = _img_device.ptr();
    render.depth = nullptr;
    render.width = resolution.width;
    render.height = resolution.height;

    render.pa.count = unsigned char(_cut_planes.size());
    std::copy(_cut_planes.begin(), _cut_planes.end(), render.pa.planes);

    Eigen::Matrix<float, 4, 4, Eigen::RowMajor>::Map(&render.inverse_modelview.m[0].x) = modelview.inverse();
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor>::Map(&render.inverse_projection.m[0].x) = projection.inverse();

    render.region = region;

    // copy over the parameters
    _ra_device.write(&ra);

    // set up kernel
    dim3 threads;
    threads.x = 16;
    threads.y = 16;
    threads.z = 1;
    dim3 blocks;
    blocks.x = static_cast<unsigned int>(std::ceil(resolution.width / (float)threads.x));
    blocks.y = static_cast<unsigned int>(std::ceil(resolution.height / (float)threads.y));
    blocks.z = 1;

    // launch the render
    launch_multirender(blocks, threads, _ra_device.ptr(), _vol_tex.handle(), _tf, _rs);
    //launch_e_render(blocks, threads, (uint32_t*)_img_device.ptr(), render.width, render.height, _vol_tex.handle(), _tf.tex, _rs.edge_enhancement, _rs.retina, _tf.threshold, _rs.brightness, render.inverse_modelview, render.inverse_projection, _tf.offset, _tf.scale, 0, 0, { -1, -1, -1 }, { 1, 1, 1 }, -1, 1, -1, 1);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        throw std::runtime_error(fmt::format("unable to launch kernel: {}", cudaGetErrorName(error)));
    }

    // read back image
    _img_device.read(out);
}