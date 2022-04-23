#include "readTable.h"
#include "stringFunctions.h"
#include "cudaWrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string>
#include <vector>
#include <algorithm>  
#include <map>
#include <numeric>


template<MemoryType memtype>
void performCommunication(int my_rank, 
        std::vector<size_t> &vec_Bytes, 
        std::vector<int> &vec_RecvRanks,
        std::vector<int> &vec_Tags){

    std::string prefix = sjoin("[ Rank ", my_rank, " ]: ");

    size_t sumOfBytes = 0;
    for(auto& bytes : vec_Bytes) sumOfBytes += bytes;
    
    SimpleMemory<memtype> SendBuffer(sumOfBytes);
    SimpleMemory<memtype> RecvBuffer(sumOfBytes);

    
    std::vector<MPI_Request> SendRequests;
    std::vector<MPI_Request> RecvRequests;

    SendRequests.resize(vec_Bytes.size());
    RecvRequests.resize(vec_Bytes.size());


    printLine(prefix,"Start communication...");
    size_t offset = 0;
    for(int i = 0; i < vec_Bytes.size(); i++)
    {
        
        printLine(prefix, "Send ", vec_Bytes[i], " bytes from rank ", my_rank, 
                " to rank ", vec_RecvRanks[i], " with tag ", vec_Tags[i]);

        MPI_Isend(SendBuffer.getPtr() + offset, vec_Bytes[i], MPI_CHAR, 
                                                vec_RecvRanks[i], vec_Tags[i], 
                                                MPI_COMM_WORLD, &SendRequests[i]);

        MPI_Irecv(RecvBuffer.getPtr() + offset, vec_Bytes[i], MPI_CHAR, 
                                                vec_RecvRanks[i], vec_Tags[i], 
                                                MPI_COMM_WORLD, &RecvRequests[i]);
        offset += vec_Bytes[i];
    }
    MPI_Waitall(vec_Bytes.size(), &RecvRequests[0], MPI_STATUS_IGNORE);
    MPI_Waitall(vec_Bytes.size(), &SendRequests[0], MPI_STATUS_IGNORE);

    for(auto& req : SendRequests){
        if ((req != MPI_REQUEST_NULL) && (req != 0)) {
            MPI_Request_free(&req);
        }
    }
    for(auto& req : RecvRequests){
        if ((req != MPI_REQUEST_NULL) && (req != 0)) {
            MPI_Request_free(&req);
        }
    }

    printLine(prefix,"Communication done!");
}



int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);


    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int my_rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    std::string prefix = sjoin("[ Rank ", my_rank, " ]: ");
    
    if(argc != 3) {
        if (my_rank == 0){
            printLine("Please specify the path and onDevice!\n",
            "Usage:\n",
            "\t mpiexec -n <processCount> ./cudaAwareMPITest <path> <onDevice>\n\n",
            "<processCount>\t: Number of processes.\n",
            "<path>\t\t: Path to communication table.\n",
            "<onDevice>\t: \"true\" or \"false\". Specify whether to communicate gpu or cpu memory.\n"
            );
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::vector<size_t> vec_Bytes;
    std::vector<int> vec_Tags;
    std::vector<int> vec_RecvRanks;
    
    std::string path = argv[1];
    std::string onDevice = argv[2];
    readTable(path, my_rank, world_size, vec_Bytes, vec_RecvRanks, vec_Tags);
    

    if(onDevice == "false"){
        performCommunication<host>(my_rank, vec_Bytes, vec_RecvRanks, vec_Tags);
    }
    else if(onDevice == "true"){
        performCommunication<device>(my_rank, vec_Bytes, vec_RecvRanks, vec_Tags);
    }
    else{
        if (my_rank == 0){
            printLine(COLORS::red,"\"", onDevice, "\" is not allowed as <onDevice> parameter!",COLORS::reset);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}

