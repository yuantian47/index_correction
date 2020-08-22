#pragma once
#ifndef RAYCAST_CUDA_HPP
#define RAYCAST_CUDA_HPP

#include <cuda_runtime.h>
#include <utility>

template<typename T>
class DeviceArray;

class Stream;

class Event {
public:
    Event(unsigned int flags = cudaEventDefault);
    ~Event();

    void sync() const;
    bool done() const;

    void record();
    void record(const Stream& stream);

    // compute time in second from start to this event
    float elapsed(const Event& start) const;

    cudaEvent_t handle() const;

protected:
    cudaEvent_t _event;
};

class Texture {
public:
    Texture();

    template<typename T>
    Texture(const DeviceArray<T>& array, const cudaTextureDesc& texture_desc) : Texture() {
        cudaResourceDesc resource_desc;
        std::memset(&resource_desc, 0, sizeof(resource_desc));

        resource_desc.resType = cudaResourceTypeArray;
        resource_desc.res.array.array = array.array();

        cudaError_t error = cudaCreateTextureObject(&_texture, &resource_desc, &texture_desc, 0);
        if(error) {
            throw std::runtime_error(fmt::format("unable to create texture object: {}", cudaGetErrorName(error)));
        }
    }

    ~Texture();

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Texture&& o);
    Texture& operator=(Texture&& o);

    bool valid() const;

    void reset();

    cudaTextureObject_t handle() const;

protected:
    cudaTextureObject_t _texture;
};

class Surface {
public:
    Surface();

    template<typename T>
    Surface(const DeviceArray<T>& array) : Surface() {
        cudaResourceDesc resource_desc;
        std::memset(&resource_desc, 0, sizeof(resource_desc));

        resource_desc.resType = cudaResourceTypeArray;
        resource_desc.res.array.array = array.array();

        cudaError_t error = cudaCreateSurfaceObject(&_surface, &resource_desc);
        if(error) {
            throw std::runtime_error(fmt::format("unable to create surface object: {}", cudaGetErrorName(error)));
        }
    }

    ~Surface();

    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    Surface(Surface&& o);
    Surface& operator=(Surface&& o);

    bool valid() const;

    cudaSurfaceObject_t handle() const;

protected:
    cudaSurfaceObject_t _surface;
};

class Stream {
public:
    Stream(unsigned int flags = cudaStreamDefault);

    ~Stream();

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& o);
    Stream& operator=(Stream&& o);

    cudaStream_t handle() const;

    void sync() const;

    void wait(const Event& event) const;

    bool ready() const;

protected:
    cudaStream_t _stream;
};

template<typename T>
class DeviceArray {
public:

    DeviceArray(const Stream* stream = nullptr)
        : _stream(stream) {
        _stream = stream;

        _reset();
    }

    DeviceArray(size_t width, size_t height = 0, size_t depth = 0, const Stream* stream = nullptr) : DeviceArray(stream) {
        allocate(width, height, depth);
    }

    DeviceArray(cudaExtent extent, const Stream* stream = nullptr) : DeviceArray(stream) {
        allocate(extent);
    }

    DeviceArray(dim3 dims, const Stream* stream = nullptr) : DeviceArray(stream) {
        allocate(dims);
    }

    ~DeviceArray() {
        release();
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& o) : DeviceArray() {
        *this = std::move(o);
    }
    DeviceArray& operator=(DeviceArray&& o) {
        release();

        std::swap(_array, o._array);
        std::swap(_extent, o._extent);

        std::swap(_stream, o._stream);

        o._reset();

        return *this;
    }

    bool valid() const {
        return _array != nullptr;
    }

    cudaArray_t array() const {
        return _array;
    }

    cudaExtent extent() const {
        return _extent;
    }

    size_t count() const {
        size_t n = _extent.width;
        if(_extent.height > 0) {
            n *= _extent.height;
        }
        if(_extent.depth > 0) {
            n *= _extent.depth;
        }
        return n;
    }

    void release() {
        cudaError_t error;

        if(_array) {
            error = cudaFreeArray(_array);
            if(error) {
                throw std::runtime_error(fmt::format("unable to free device array: {}", cudaGetErrorName(error)));
            }
        }

        _reset();
    }

    void allocate(size_t width, size_t height = 0, size_t depth = 0) {
        return allocate(make_cudaExtent(width, height, depth));
    }

    void allocate(dim3 dims) {
        return allocate(make_cudaExtent(dims.x, dims.y, dims.z));
    }

    void allocate(cudaExtent extent) {
        release();

        _extent = extent;
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();

        cudaError_t error = cudaMalloc3DArray(&_array, &desc, _extent, cudaArraySurfaceLoadStore);
        if(error) {
            throw std::runtime_error(fmt::format("unable to allocate device array on stream {}: {}", (size_t)_stream, cudaGetErrorName(error)));
        }
    }

    void write(const T* ptr, const Stream* stream = nullptr) {
        return write(ptr, _extent.width * sizeof(T));
    }

    void write(const T* ptr, size_t pitch, const Stream* stream = nullptr) {
        cudaPos offset = { 0 };
        return write(ptr, pitch, offset, _extent);
    }

    void write(const T* ptr, size_t pitch, const cudaPos& offset, const cudaExtent& extent, const Stream* stream = nullptr) {
        cudaMemcpy3DParms params;
        std::memset(&params, 0, sizeof(params));

        params.srcPtr = make_cudaPitchedPtr(const_cast<T*>(ptr), pitch, _extent.width, _extent.height);
        params.dstArray = _array;
        params.dstPos = offset;
        params.extent = extent;
        params.kind = cudaMemcpyHostToDevice;

        _memcpy(params, stream);
    }

    void read(T* ptr) {
        return read(ptr, _extent.width * sizeof(T));
    }

    void read(T* ptr, size_t pitch, const Stream* stream = nullptr) {
        cudaPos offset = { 0 };
        return read(ptr, pitch, offset, _extent);
    }

    void read(T* ptr, size_t pitch, const cudaPos& offset, const cudaExtent& extent, const Stream* stream = nullptr) {
        cudaMemcpy3DParms params;
        std::memset(&params, 0, sizeof(params));

        params.srcArray = _array;
        params.srcPos = offset;
        params.dstPtr = make_cudaPitchedPtr(const_cast<T*>(ptr), pitch, _extent.width, _extent.height);
        params.extent = extent;
        params.kind = cudaMemcpyDeviceToHost;

        _memcpy(params, stream);
    }

    Texture texture(cudaTextureAddressMode address = cudaAddressModeClamp, cudaTextureFilterMode filter = cudaFilterModeLinear, cudaTextureReadMode read = cudaReadModeNormalizedFloat) const {
        cudaTextureDesc texture_desc;
        std::memset(&texture_desc, 0, sizeof(texture_desc));

        texture_desc.addressMode[0] = address;
        texture_desc.addressMode[1] = address;
        texture_desc.addressMode[2] = address;
        texture_desc.filterMode = filter;
        texture_desc.readMode = read;
        texture_desc.normalizedCoords = 1;

        return texture(texture_desc);
    }

    Texture texture(const cudaTextureDesc& texture_desc) const {
        return Texture(*this, texture_desc);
    }

    Surface surface() const {
        return Surface(*this);
    }

protected:
    void _memcpy(cudaMemcpy3DParms& params, const Stream* stream) {
        // the memcpy must have non-zero height and depth for 1D and 2D arrays
        if(params.extent.height == 0) {
            params.extent.height = 1;
        }
        if(params.extent.depth == 0) {
            params.extent.depth = 1;
        }

        cudaStream_t sid;
        if(stream) {
            sid = stream->handle();
        } else if(_stream) {
            sid = _stream->handle();
        } else {
            sid = (cudaStream_t)0;
        }

        cudaError_t error = cudaMemcpy3DAsync(&params, sid);
        if(error) {
            throw std::runtime_error(fmt::format("unable to memcpy to/from device array on stream {}: {}", (size_t)sid, cudaGetErrorName(error)));
        }
    }

    void _reset() {
        _array = nullptr;
        _extent = { 0 };
    }

    cudaArray_t _array;
    cudaExtent _extent;

    const Stream* _stream;
};

template<typename T>
class DeviceMemory {
public:

    DeviceMemory(const Stream* stream = nullptr)
        : _stream(stream) {

        _reset();
    }

    DeviceMemory(size_t count, const Stream* stream = nullptr) : DeviceMemory(stream) {
        allocate(count);
    }

    ~DeviceMemory() {
        release();
    }

    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    DeviceMemory(DeviceMemory&& o) : DeviceMemory() {
        *this = std::move(o);
    }
    DeviceMemory& operator=(DeviceMemory&& o) {
        release();
        _reset();

        std::swap(_ptr, o._ptr);
        std::swap(_count, o._count);

        std::swap(_stream, o._stream);

        return *this;
    }

    bool valid() const {
        return _ptr != nullptr;
    }

    size_t count() const {
        return _count;
    }

    size_t size() const {
        return _count * sizeof(T);
    }

    T* ptr() const {
        return _ptr;
    }

    void release() {
        cudaError_t error;

        if(_ptr) {
            error = cudaFree(_ptr);
            if(error) {
                throw std::runtime_error(fmt::format("unable to free device memory: {}", cudaGetErrorName(error)));
            }
        }

        _reset();
    }

    void allocate(size_t count) {
        release();

        _count = count;

        cudaError_t error = cudaMalloc(&_ptr, size());
        if(error) {
            throw std::runtime_error(fmt::format("unable to allocate device memory: {}", cudaGetErrorName(error)));
        }
    }

    void write(const T* host, const Stream* stream = nullptr) {
        return write(host, 0, size());
    }

    void write(const T* host, size_t offset, size_t size, const Stream* stream = nullptr) {
        _memcpy(_ptr + offset, host, size, cudaMemcpyHostToDevice, stream);
    }

    void read(T* host, const Stream* stream = nullptr) {
        return read(host, 0, size());
    }

    void read(T* host, size_t offset, size_t size, const Stream* stream = nullptr) {
        _memcpy(host, _ptr + offset, size, cudaMemcpyDeviceToHost, stream);
    }

protected:
    void _memcpy(T* dst, const T* src, size_t count, cudaMemcpyKind kind, const Stream* stream) {
        cudaStream_t sid;
        if(stream) {
            sid = stream->handle();
        } else if(_stream) {
            sid = _stream->handle();
        } else {
            sid = (cudaStream_t)0;
        }

        cudaError_t error = cudaMemcpyAsync(dst, src, count, kind, sid);
        if(error) {
            throw std::runtime_error(fmt::format("unable to memcpy to/from device memory on stream {}: {}", (size_t)sid, cudaGetErrorName(error)));
        }
    }

    void _reset() {
        _ptr = nullptr;
        _count = 0;
    }

    T* _ptr;
    size_t _count;

    const Stream* _stream;
};

template<typename T>
class PinnedMemory {
public:

    PinnedMemory() {
        _reset();
    }

    PinnedMemory(size_t count) : PinnedMemory() {
        allocate(count);
    }

    ~PinnedMemory() {
        release();
    }

    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;

    PinnedMemory(PinnedMemory&& o) : PinnedMemory() {
        *this = std::move(o);
    }
    PinnedMemory& operator=(PinnedMemory&& o) {
        release();
        _reset();

        std::swap(_ptr, o._ptr);
        std::swap(_count, o._count);

        return *this;
    }

    bool valid() const {
        return _ptr != nullptr;
    }

    size_t count() const {
        return _count;
    }

    size_t size() const {
        return _count * sizeof(T);
    }

    T* ptr() const {
        return _ptr;
    }

    void release() {
        cudaError_t error;

        if(_ptr) {
            error = cudaFreeHost(_ptr);
            if(error) {
                throw std::runtime_error(fmt::format("unable to free pinned memory: {}", cudaGetErrorName(error)));
            }
        }

        _reset();
    }

    void allocate(size_t count) {
        release();

        _count = count;

        cudaError_t error = cudaMallocHost(&_ptr, size());
        if(error) {
            throw std::runtime_error(fmt::format("unable to allocate pinned memory: {}", cudaGetErrorName(error)));
        }
    }

protected:
    void _reset() {
        _ptr = nullptr;
        _count = 0;
    }

    T* _ptr;
    size_t _count;
};

#endif