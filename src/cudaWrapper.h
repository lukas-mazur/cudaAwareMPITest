#ifndef CUDAWRAPPER
#define CUDAWRAPPER

#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>
#include <random>
#include <limits>
#include <vector>
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
        dtype *_buffer;
        size_t _size;
        std::string _prefix;

    public:


        SimpleMemory(size_t size, std::string prefix = "") {
            _size = size;
            _prefix = prefix;
            if(_size != 0) {
                switch(memType) {
                    case device:
                        printLine(_prefix, "Allocating ", _size, " bytes on device");
                        checkCudaErrors(cudaMalloc((void**)&_buffer, size));
                        break;

                    case hostPinned:
                        printLine(_prefix, "Allocating ", _size, " bytes on host (pinned)");
                        checkCudaErrors(cudaMallocHost((void**)&_buffer, size));
                        break;

                    case host:
                        printLine(_prefix, "Allocating ", _size, " bytes on host");
                        _buffer = static_cast<dtype *>(std::malloc(size));
                        if (_buffer == NULL) {
                            printLine(_prefix, "Error creating host buffer!");
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
                            printLine(_prefix, "Free device memory");
                            checkCudaErrors(cudaFree(_buffer));
                            break;
                        case hostPinned:
                            printLine(_prefix, "Free pinned host memory");
                            checkCudaErrors(cudaFreeHost(_buffer));
                            break;
                        case host:
                            printLine(_prefix, "Free host memory");
                            std::free(_buffer);
                            break;
                    }
            }
        }

        dtype *getPtr() { return _buffer; }

        size_t getSize() { return _size; }

        void insert(dtype* data, size_t offset, size_t size){
            if(offset + size > _size){
                printLine(COLORS::red, _prefix, "Error! SimpleMemory<dtype>::insert(..), offset+size exceeds array boundary!", COLORS::reset);
                exit(-1);
            }

            if(memType == host || memType == hostPinned) {
                std::memcpy(_buffer+offset, data, size);
            }
            else {
                checkCudaErrors(cudaMemcpy(_buffer+offset, data, size, cudaMemcpyHostToDevice));
            }
        }

};


template <MemoryType memType1, MemoryType memType2, typename dtype>
inline void copyMemory(SimpleMemory<memType1, dtype> &src,
        SimpleMemory<memType2, dtype> &dst) {

    if ((memType1 == host || memType1 == hostPinned) &&
             (memType2 == host || memType2 == hostPinned)){
        std::memcpy(dst.getPtr(), src.getPtr(), src.getSize());
        return;
    }

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

    if (src.getSize() != dst.getSize()){
        printLine("Error! Memcpy: src.getSize() != dst.getSize()");
        exit(EXIT_FAILURE);
    }
    checkCudaErrors(cudaMemcpy(dst.getPtr(), src.getPtr(), src.getSize(), kind));
}

#endif
