#ifndef CUDAWRAPPER
#define CUDAWRAPPER

#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>
#include "stringFunctions.h"

template <typename T>
inline void check(T result, char const *const func, const char *const file,
        int const line) {
    if (result) {
        printFormat("CUDA error at %s:%d code=%d(%s) \"%s\"", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


template <typename dtype = char> inline size_t Bytes(double size) { 
    return size_t(size / sizeof(dtype)); 
}
template <typename dtype = char> inline size_t KiB(double size) {
    return size_t((std::pow(2, 10) * size) / sizeof(dtype));
}
template <typename dtype = char> inline size_t MiB(double size) {
    return size_t((std::pow(2, 20) * size) / sizeof(dtype));
}
template <typename dtype = char> inline size_t GiB(double size) {
    return size_t((std::pow(2, 30) * size) / sizeof(dtype));
}


enum MemoryType { device, host, hostPinned };

template <MemoryType memType = host, typename dtype = char> 
class SimpleMemory {
    private:

    public:
        dtype *_buffer;
        size_t _size;


        SimpleMemory(size_t size) {
            _size = size;
            if(_size != 0) {
                switch(memType) {
                    case device:
                        printFormat("Allocating %zu bytes on device", size);
                        checkCudaErrors(cudaMalloc((void**)&_buffer, size));
                        break;

                    case hostPinned:
                        printFormat("Allocating %zu bytes on host (pinned)", size);
                        checkCudaErrors(cudaMallocHost((void**)&_buffer, size));
                        break;

                    case host:
                        printFormat("Allocating %zu bytes on host", size);
                        _buffer = static_cast<dtype *>(std::malloc(size));
                        if (_buffer == NULL) {
                            printFormat("Error creating host buffer!");
                            exit(EXIT_FAILURE);
                        }
                        break;
                }
            }
        }

        ~SimpleMemory() {
            if(_size != 0) {
                switch(memType) {
                        case device:
                            printLine("Free device memory");
                            checkCudaErrors(cudaFree(_buffer));
                            break;
                        case hostPinned:
                            printLine("Free pinned host memory");
                            checkCudaErrors(cudaFreeHost(_buffer));
                            break;
                        case host:
                            printLine("Free host memory");
                            std::free(_buffer);
                            break;
                    }
            }
        }

        dtype *getPtr() { return _buffer; }

        size_t getSize() { return _size; }
};


template <MemoryType memType1, MemoryType memType2, typename dtype>
inline cudaError_t copyMemory(SimpleMemory<memType1, dtype> &src,
        SimpleMemory<memType2, dtype> &dst) {

    cudaMemcpyKind kind;

    if (memType1 == device && memType2 == device){
        kind = cudaMemcpyDeviceToDevice;
    }
    else if (memType1 == device && (memType2 == host || memType2 == hostPinned)){
        kind = cudaMemcpyDeviceToHost;
    }
    else if ((memType1 == host || memType1 == hostPinned) && memType2 == device){
        kind = cudaMemcpyHostToDevice;
    }
    else if ((memType1 == host || memType1 == hostPinned) &&
             (memType2 == host || memType2 == hostPinned)){
        kind = cudaMemcpyHostToHost;
    }

    if (src.getSize() != dst.getSize()){
        printLine("Error! Memcpy: src.getSize() != dst.getSize()");
        exit(EXIT_FAILURE);
    }
    return cudaMemcpy(dst.getPtr(), src.getPtr(), src.getSize(), kind);
}

#endif
