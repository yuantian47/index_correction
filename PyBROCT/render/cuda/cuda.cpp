#include "cuda.hpp"

#include <fmt/format.h>

Event::Event(unsigned int flags) {
    cudaError_t error = cudaEventCreateWithFlags(&_event, flags);
    if(error) {
        throw std::runtime_error(fmt::format("unable to create event: {}", cudaGetErrorName(error)));
    }
}

Event::~Event() {
    cudaError_t error = cudaEventDestroy(_event);
    if(error) {
        //logger.warn("unable to destroy event: {} -> ignoring", cudaGetErrorName(error));
    }
}

void Event::sync() const {
    cudaError_t error = cudaEventSynchronize(_event);
    if(error) {
        throw std::runtime_error(fmt::format("unable to synchronize event {}: {}", (size_t)_event, cudaGetErrorName(error)));
    }
}

bool Event::done() const {
    cudaError_t error = cudaEventQuery(_event);
    if(error == cudaErrorNotReady) {
        return false;
    } else {
        throw std::runtime_error(fmt::format("unable to query event {}: {}", (size_t)_event, cudaGetErrorName(error)));
    }

    return true;
}

void Event::record() {
    cudaError_t error = cudaEventRecord(_event);
    if(error) {
        throw std::runtime_error(fmt::format("unable to record event {}: {}", (size_t)_event, cudaGetErrorName(error)));
    }
}

void Event::record(const Stream& stream) {
    cudaError_t error = cudaEventRecord(_event, stream.handle());
    if(error) {
        throw std::runtime_error(fmt::format("unable to record event {} on stream {}: {}", (size_t)_event, (size_t)stream.handle(), cudaGetErrorName(error)));
    }
}

float Event::elapsed(const Event& start) const {
    float result;
    cudaError_t error = cudaEventElapsedTime(&result, start._event, _event);
    if(error) {
        throw std::runtime_error(fmt::format("unable to compute elapsed time from event {} to event {}: {}", (size_t)start._event, (size_t)_event, cudaGetErrorName(error)));
    }

    return result * 1000;
}

cudaEvent_t Event::handle() const {
    return _event;
}

Texture::Texture() {
    _texture = 0;
}

Texture::~Texture() {
    if(valid()) {
        cudaError_t error = cudaDestroyTextureObject(_texture);
        if(error) {
            //logger.warn("unable to destroy texture object: {} -> ignoring", cudaGetErrorName(error));
        }
    }
}

Texture::Texture(Texture&& o) : Texture() {
    *this = std::move(o);
}
Texture& Texture::operator=(Texture&& o) {
    std::swap(_texture, o._texture);

    return *this;
}

bool Texture::valid() const {
    return _texture != 0;
}

void Texture::reset() {
    *this = std::move(Texture());
}

cudaTextureObject_t Texture::handle() const {
    return _texture;
}

Surface::Surface() {
    _surface = 0;
}

Surface::~Surface() {
    if(valid()) {
        cudaError_t error = cudaDestroySurfaceObject(_surface);
        if(error) {
            //logger.warn("unable to destroy surface object: {} -> ignoring", cudaGetErrorName(error));
        }
    }
}

Surface::Surface(Surface&& o) : Surface() {
    *this = std::move(o);
}
Surface& Surface::operator=(Surface&& o) {
    std::swap(_surface, o._surface);

    return *this;
}

bool Surface::valid() const {
    return _surface != 0;
}

cudaSurfaceObject_t Surface::handle() const {
    return _surface;
}

Stream::Stream(unsigned int flags) {
    cudaError_t error = cudaStreamCreateWithFlags(&_stream, flags);
    if(error) {
        throw std::runtime_error(fmt::format("unable to create stream {}: {}", (size_t)_stream, cudaGetErrorName(error)));
    }
}

Stream::~Stream() {
    if(_stream != 0) {
        cudaError_t error = cudaStreamDestroy(_stream);
        if(error) {
            //logger.warn("unable to destroy stream {}: {} -> ignoring", (size_t)_stream, cudaGetErrorName(error));
        }
    }
}

Stream::Stream(Stream&& o) : Stream() {
    *this = std::move(o);
}
Stream& Stream::operator=(Stream&& o) {
    std::swap(_stream, o._stream);

    return *this;
}

cudaStream_t Stream::handle() const {
    return _stream;
}

void Stream::sync() const {
    cudaError_t error = cudaStreamSynchronize(_stream);
    if(error) {
        throw std::runtime_error(fmt::format("unable to synchronize stream {}: {}", (size_t)_stream, cudaGetErrorName(error)));
    }
}

void Stream::wait(const Event& event) const {
    cudaError_t error = cudaStreamWaitEvent(_stream, event.handle(), 0);
    if(error) {
        throw std::runtime_error(fmt::format("unable to wait for event {} on stream {}: {}", (size_t)event.handle(), (size_t)_stream, cudaGetErrorName(error)));
    }
}

bool Stream::ready() const {
    cudaError_t error = cudaStreamQuery(_stream);
    if(error == cudaErrorNotReady) {
        return false;
    } else {
        throw std::runtime_error(fmt::format("unable to query stream {}: {}", (size_t)_stream, cudaGetErrorName(error)));
    }

    return true;
}
