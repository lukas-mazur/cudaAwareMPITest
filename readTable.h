
#include "csvstream.h"
#include "stringFunctions.h"
#include <mpi.h>


void readTable(std::string path, int my_rank, int world_size,
        std::vector<size_t> &vec_Bytes, 
        std::vector<int> &vec_SendRanks,
        std::vector<int> &vec_RecvRanks)
{

    csvstream csvin(path);

    std::string prefix = sjoin("[ Rank ", my_rank, " ]: ");
    // Rows have key = column name, value = cell datum
    std::map<std::string, std::string> row;
    
    std::vector<size_t> vec_AllBytes;
    std::vector<int> vec_AllSendRanks;
    std::vector<int> vec_AllRecvRanks;

    // Read communication table
    while (csvin >> row) {
        std::stringstream sstream_Bytes(row["Bytes"]);
        size_t result_Bytes;
        sstream_Bytes >> result_Bytes;
        vec_AllBytes.push_back(result_Bytes);

        std::stringstream sstream_SendRank(row["SendRank"]);
        int result_SendRank;
        sstream_SendRank >> result_SendRank;
        vec_AllSendRanks.push_back(result_SendRank);

        std::stringstream sstream_RecvRank(row["RecvRank"]);
        int result_RecvRank;
        sstream_RecvRank >> result_RecvRank;
        vec_AllRecvRanks.push_back(result_RecvRank);
    }

    std::vector<size_t> vec_tmp;
    vec_tmp.insert(vec_tmp.end(), vec_AllSendRanks.begin(), vec_AllSendRanks.end());
    vec_tmp.insert(vec_tmp.end(), vec_AllRecvRanks.begin(), vec_AllRecvRanks.end());
    size_t max_rank = *max_element(vec_tmp.begin(), vec_tmp.end());


    if(world_size != (max_rank + 1)){
        printLine(prefix, "communicationTable.csv requires this application to be run with ", max_rank + 1," processes.");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::vector<size_t> vec_myIndices;

    size_t index = 0;
    for (auto& entry : vec_AllSendRanks)
    {
        if (entry == my_rank) vec_myIndices.push_back(index);
        index++;
    }
    
    index = 0;
    for (auto& entry : vec_AllRecvRanks)
    {
        if (entry == my_rank) vec_myIndices.push_back(index);
        index++;
    }
    
    sort(vec_myIndices.begin(), vec_myIndices.end());
    
    for(auto& i : vec_myIndices){
        vec_Bytes.push_back(vec_AllBytes[i]);
        vec_SendRanks.push_back(vec_AllSendRanks[i]);
        vec_RecvRanks.push_back(vec_AllRecvRanks[i]);
    }
}
