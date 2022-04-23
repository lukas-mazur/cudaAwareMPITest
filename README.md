# cudaAwareMPITest
This program tests asynchronous cuda-aware MPI calls on any number of processes.

## Prerequisites
Download and install CUDA and a cuda-aware MPI implementation.

## Getting cudaAwareMPITest

Using git clone the repository of cudaAwareMPITest using the command below.
```
git clone https://github.com/lukas-mazur/cudaAwareMPITest.git
```

## Build cudaAwareMPITest
The code is built using cmake. Change the current directory to the directory of the repository and use the command below to build the code.
```
mkdir build && cd build
cmake ../
make
```
