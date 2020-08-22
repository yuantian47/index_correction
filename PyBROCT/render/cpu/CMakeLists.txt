# ref: http://www.benjack.io/2018/02/02/python-cpp-revisited.html
cmake_minimum_required(VERSION 3.0)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake")

project(PyBROCT_render_cpu LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(cpu SHARED
  raycast.cpp raycast.hpp
  filter.cpp filter.hpp
  render.cpp render.hpp
  bindings.cpp
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(cpu PRIVATE Eigen3::Eigen)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(cpu PRIVATE fmt::fmt)
