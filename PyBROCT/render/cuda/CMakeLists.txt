# ref: http://www.benjack.io/2018/02/02/python-cpp-revisited.html
cmake_minimum_required(VERSION 3.0)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake")

project(PyBROCT_render_cuda LANGUAGES CXX CUDA)

add_library(cuda SHARED
  raycast.cu raycast.cuh
  filter.cu filter.cuh
  render.cpp render.hpp
  cuda.cpp cuda.hpp
  helper_math.hpp
  bindings.cpp
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61 -lineinfo -Xptxas -v")

# ref: https://github.com/torstem/demo-cuda-pybind11/blob/master/CMakeLists.txt
include(FindPythonLibsNew)
target_include_directories(cuda PRIVATE ${SOURCE_DIR} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(cuda PRIVATE ${PYTHON_LIBRARIES})

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(cuda PRIVATE Eigen3::Eigen)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(cuda PRIVATE fmt::fmt)

set_target_properties(cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)