
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(cudaAwareMPITest LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)

find_package(MPI 3.1 REQUIRED)
find_package(CUDA 11.0 REQUIRED)

include_directories(SYSTEM "${MPI_INCLUDE_PATH};${CUDA_INCLUDE_DIRS}")
link_libraries("${MPI_LIBRARIES};${CUDA_LIBRARIES}")

add_executable(cudaAwareMPITest src/main.cpp)

