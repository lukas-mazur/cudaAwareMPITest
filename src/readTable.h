
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
    std::map<std::string, std::string> row;

    std::vector<size_t> vec_AllBytes;
    std::vector<int> vec_AllRanksA;
    std::vector<int> vec_AllRanksB;

    // Read communication table
    while (csvin >> row) {
        std::stringstream sstream_Bytes(row["Bytes"]);
        size_t result_Bytes;
        sstream_Bytes >> result_Bytes;
        vec_AllBytes.push_back(result_Bytes);

        std::stringstream sstream_SendRank(row["RankA"]);
        int result_SendRank;
        sstream_SendRank >> result_SendRank;
        vec_AllRanksA.push_back(result_SendRank);

        std::stringstream sstream_RecvRank(row["RankB"]);
        int result_RecvRank;
        sstream_RecvRank >> result_RecvRank;
        vec_AllRanksB.push_back(result_RecvRank);
    }

    std::vector<size_t> vec_tmp;
    vec_tmp.insert(vec_tmp.end(), vec_AllRanksA.begin(), vec_AllRanksA.end());
    vec_tmp.insert(vec_tmp.end(), vec_AllRanksB.begin(), vec_AllRanksB.end());
    size_t max_rank = *max_element(vec_tmp.begin(), vec_tmp.end());


    if(world_size != (max_rank + 1)){
        printLine(prefix, "communicationTable.csv requires this application to be run with ", 
                max_rank + 1," processes.");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::vector<size_t> vec_myIndices;

    size_t index = 0;
    for (auto& entry : vec_AllRanksA)
    {
        if (entry == my_rank) vec_myIndices.push_back(index);
        index++;
    }

    index = 0;
    for (auto& entry : vec_AllRanksB)
    {
        if (entry == my_rank) vec_myIndices.push_back(index);
        index++;
    }

    sort(vec_myIndices.begin(), vec_myIndices.end());

    for(auto& i : vec_myIndices){

        vec_Bytes.push_back(vec_AllBytes[i]);

        if(vec_AllRanksA[i] == my_rank){
            vec_SendRanks.push_back(vec_AllRanksA[i]);
            vec_RecvRanks.push_back(vec_AllRanksB[i]);
        }
        else{
            vec_SendRanks.push_back(vec_AllRanksB[i]);
            vec_RecvRanks.push_back(vec_AllRanksA[i]);
        }
    }
}
