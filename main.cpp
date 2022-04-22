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



void performCommunication(int my_rank, 
        std::vector<size_t> &vec_Bytes, 
        std::vector<int> &vec_SendRanks,
        std::vector<int> &vec_RecvRanks){

    std::string prefix = sjoin("[ Rank ", my_rank, " ]: ");

    size_t sumOfBytes = 0;
    for(auto& bytes : vec_Bytes) sumOfBytes += bytes;
    
    constexpr MemoryType memtype = host;

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
        
        printLine(prefix, "Send ", vec_Bytes[i],
                                 " bytes from rank ", vec_SendRanks[i], " to rank ", vec_RecvRanks[i]);

        MPI_Isend(SendBuffer.getPtr() + offset, vec_Bytes[i], MPI_CHAR, 
                                                vec_RecvRanks[i], vec_SendRanks[i], 
                                                MPI_COMM_WORLD, &SendRequests[i]);

        MPI_Irecv(RecvBuffer.getPtr() + offset, vec_Bytes[i], MPI_CHAR, 
                                                vec_RecvRanks[i], vec_RecvRanks[i], 
                                                MPI_COMM_WORLD, &RecvRequests[i]);
        offset += vec_Bytes[i];
    }
    printLine(prefix,"MPI_Waitall...");
    MPI_Waitall(vec_Bytes.size(), &RecvRequests[0], MPI_STATUS_IGNORE);
    MPI_Waitall(vec_Bytes.size(), &SendRequests[0], MPI_STATUS_IGNORE);

    printLine(prefix,"MPI_Request_free...");
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


    std::vector<size_t> vec_Bytes;
    std::vector<int> vec_SendRanks;
    std::vector<int> vec_RecvRanks;
    
    std::string path = "communicationTable.csv";
    readTable(path, my_rank, world_size, vec_Bytes, vec_SendRanks, vec_RecvRanks);
    
 //   std::string prefix = sjoin("[ Rank ", my_rank, " ]: ");
 //   for(int i = 0; i < vec_Bytes.size(); i++)
 //   {
 //       printLine(prefix, vec_Bytes[i], ",", vec_SendRanks[i], ",", vec_RecvRanks[i]);
 //   }

 //   printLine("==========================================\n");
    performCommunication(my_rank, vec_Bytes, vec_SendRanks, vec_RecvRanks);

    MPI_Finalize();

    return EXIT_SUCCESS;
}

