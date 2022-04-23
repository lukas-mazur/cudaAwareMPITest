
#include "csvstream.h"
#include "stringFunctions.h"
#include <mpi.h>


void readTable(std::string path, int my_rank, int world_size,
        std::vector<size_t> &vec_Bytes, 
        std::vector<int> &vec_RecvRanks,
        std::vector<int> &vec_Tags)
{

    csvstream csvin(path);

    std::string prefix = sjoin("[ Rank ", my_rank, " ]: ");
    std::map<std::string, std::string> row;

    std::vector<size_t> vec_AllBytes;
    std::vector<int> vec_AllRanksA;
    std::vector<int> vec_AllRanksB;
    std::vector<int> vec_AllTags;

    int tag = 0;
    // Read communication table
    while (csvin >> row) {
        std::stringstream sstream_Bytes(row["Bytes"]);
        size_t result_Bytes;
        sstream_Bytes >> result_Bytes;
        vec_AllBytes.push_back(result_Bytes);

        std::stringstream sstream_RankA(row["RankA"]);
        int result_RankA;
        sstream_RankA >> result_RankA;
        vec_AllRanksA.push_back(result_RankA);

        std::stringstream sstream_RankB(row["RankB"]);
        int result_RankB;
        sstream_RankB >> result_RankB;
        vec_AllRanksB.push_back(result_RankB);

        vec_AllTags.push_back(tag);
        tag++;
    }

    std::vector<size_t> vec_tmp;
    vec_tmp.insert(vec_tmp.end(), vec_AllRanksA.begin(), vec_AllRanksA.end());
    vec_tmp.insert(vec_tmp.end(), vec_AllRanksB.begin(), vec_AllRanksB.end());
    size_t max_rank = *max_element(vec_tmp.begin(), vec_tmp.end());


    if(world_size != (max_rank + 1)){
        printLine(COLORS::red,prefix, path, " requires this application to run with ", 
                max_rank + 1," processes.",COLORS::reset);
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
        vec_Tags.push_back(vec_AllTags[i]);

        if(vec_AllRanksA[i] == my_rank){
            vec_RecvRanks.push_back(vec_AllRanksB[i]);
        }
        else{
            vec_RecvRanks.push_back(vec_AllRanksA[i]);
        }
    }
}
