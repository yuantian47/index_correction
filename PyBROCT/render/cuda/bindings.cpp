#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cuda.hpp"
#include "render.hpp"
#include "helper_math.hpp"

namespace py = pybind11;

template<typename T>
using pair_t = std::pair<T, T>;
//struct pair_t : std::pair<T, T> {
//    using std::pair<T, T>::pair;
//
//    const T& x() const {
//        return first;
//    }
//    const T& y() const {
//        return second;
//    }
//};

template<typename T>
using triple_t = std::tuple<T, T, T>;
//struct triple_t : std::tuple<T, T, T> {
//    using std::tuple<T, T, T>::tuple;
//
//    const T& x() const {
//        return std::get<0>(*this);
//    }
//    const T& y() const {
//        return std::get<1>(*this);
//    }
//    const T& z() const {
//        return std::get<2>(*this);
//    }
//};

void _set_volume_helper(CudaRenderer& r, py::array_t<int8_t, py::array::c_style | py::array::forcecast>& data, bool filter) {
    if(data.ndim() != 3) {
        throw std::runtime_error(fmt::format("volume data must have 3 dimensions, not {}", data.ndim()));
    }

    auto info = data.request();
    {
        py::gil_scoped_release release;
        r.set_volume((const int8_t*)info.ptr, { (size_t)info.shape[0], (size_t)info.shape[1], (size_t)info.shape[2] }, filter);
    }
}

void _set_transfer_function_helper(CudaRenderer& r, py::array_t<float, py::array::c_style | py::array::forcecast>& data, float offset, float scale) {
    if(data.ndim() != 1 && data.ndim() != 2) {
        throw std::runtime_error(fmt::format("volume data must have 1 or 2 dimensions, not {}", data.ndim()));
    }

    auto info = data.request();
    if(data.ndim() == 1 || (data.ndim() == 2 && info.shape[1] == 1)) {
        std::vector<float4> tf(info.shape[0]);
        for(ptrdiff_t i = 0; i < info.shape[0]; i++) {
            tf[i] = make_float4(((const float*)info.ptr)[i]);
        }

        r.set_transfer_function(tf, offset, scale);
    } else if(data.ndim() == 2) {
        if(info.shape[1] != 4) {
            throw std::runtime_error(fmt::format("2D transfer function second shape of 1 or 4, not {}", info.shape[1]));
        }

        r.set_transfer_function((const float4*)info.ptr, info.shape[0], offset, scale);
    }
}

auto _render_helper(CudaRenderer& r, const Eigen::Matrix4f& modelview, const Eigen::Matrix4f& projection, const pair_t<size_t>& resolution, const pair_t<pair_t<float>>& region) {
    auto image = py::array_t<uint8_t, py::array::c_style>({ resolution.second, resolution.first, (size_t)4 });
    auto info = image.request();

    {
        py::gil_scoped_release release;
        r.render(modelview, projection, { resolution.first, resolution.second }, { { region.first.first, region.first.second}, {region.second.first, region.second.second} }, (uchar4*)info.ptr);
    }

    return image;
}

PYBIND11_MODULE(cuda, m) {

    py::class_<CudaRenderer>(m, "CudaRenderer")
        .def(py::init<>())

        .def("set_volume", &_set_volume_helper)
        .def("set_volume", [](CudaRenderer& r, py::array_t<int8_t, py::array::c_style | py::array::forcecast>& data) { _set_volume_helper(r, data, false); })

        .def("set_threshold", &CudaRenderer::set_threshold)
        .def("set_brightness", &CudaRenderer::set_brightness)
        .def("set_range", &CudaRenderer::set_range)
        .def("set_enhancement", &CudaRenderer::set_enhancement)

        .def("set_transfer_function", &_set_transfer_function_helper)
        .def("set_transfer_function", [](CudaRenderer& r, py::array_t<float, py::array::c_style | py::array::forcecast>& data) { _set_transfer_function_helper(r, data, 0, 1); })

        .def("set_cut_planes", [](CudaRenderer& r, const std::vector<pair_t<triple_t<float>>> planes) {
        std::vector<Plane> planes2;
        for(const auto& p : planes) {
            planes2.push_back({ { std::get<0>(p.first), std::get<1>(p.first), std::get<2>(p.first) }, { std::get<0>(p.second), std::get<1>(p.second), std::get<2>(p.second) } });
        }
        r.set_cut_planes(std::move(planes2));
    })

        .def("render", &_render_helper)
        .def("render", [](CudaRenderer& r, const Eigen::Matrix4f& modelview, const Eigen::Matrix4f& projection, const pair_t<size_t>& resolution) { _render_helper(r, modelview, projection, resolution, { {-1.0f, 1.0f}, {-1.0f, 1.0f} }); })

        .def_property_readonly_static("type", [](py::object&) { return "CUDA"; })
    ;

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
