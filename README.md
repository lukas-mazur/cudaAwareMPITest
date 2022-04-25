# cudaAwareMPITest
This program tests asynchronous bidirectional cuda-aware MPI calls on any number of processes.

## Prerequisites
Download and install CUDA and a cuda-aware MPI implementation.

## Download

Clone the repository of cudaAwareMPITest using the command below.
```
git clone https://github.com/lukas-mazur/cudaAwareMPITest.git
```

## Build
The code is built using cmake. Change the current directory to the directory of the repository and use the command below to build the code.
```
mkdir build && cd build
cmake ../
make
```

## Run

Call cudaAwareMPITest as follows:
```
mpiexec -n <processCount> ./cudaAwareMPITest <path> <onDevice>
```
where you need to specify:
```
<processCount>  : Number of processes.
<path>          : Path to communication table. Some examples are available in the sub-folder communicationTables/.
<onDevice>      : Set it to "true" in order to communicate gpu memory. Set it to "false" in order to communicate cpu memory.
```

## Creating a communication table

Each row in the table describes a bidirectional communication call. The first column is the number of bytes which should be transfered, while the second and third columns are the ranks which are communicating.

In the example below two bidirectional communication calls will be performed. The first call communicates 123 Bytes between rank 0 and 1, while the second call communicates 12345 Bytes between rank 0 and 2.
```
Bytes,RankA,RankB
123,0,1
12345,0,2
```

Some more examples are available in the sub-folder `communicationTables/`.
