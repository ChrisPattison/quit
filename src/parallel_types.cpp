#include "parallel_types.hpp"
#include <cmath>

namespace parallel  {
    
int Heirarchy::rank(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

const std::vector<MPI_Comm>& Heirarchy::comms() {
    return comms_;
}

int Heirarchy::global_levels() {
    return levels_;
}

int Heirarchy::local_levels() {
    return comms_.size();
}

int Heirarchy::base() {
    return base_;
}

Heirarchy::Heirarchy() {
    levels_ = 0;
    base_ = 0;
    comms_ = { };
}

Heirarchy::Heirarchy(int base) {
    comms_ = { };
    base_ = base;
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    world_rank = rank(MPI_COMM_WORLD);

    MPI_Group mpi_world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &mpi_world_group);

    int level_stride = 1; // level_stride = base ^ level
    levels_ = std::ceil(std::log(world_size)/std::log(base_));
    for(int level = 0; world_rank % level_stride == 0 && level < levels_; ++level, level_stride *= base_) {
        MPI_Comm level_comm;
        MPI_Comm next_comm;
        MPI_Group level_group;
        int kmax = world_size / level_stride;

        // Build group
        std::vector<int> group_indices;
        group_indices.reserve(kmax);
        for(int k = 0; k <= kmax; ++k) {
            int rank = k*level_stride;
            if(rank < world_size) {
                group_indices.push_back(rank);
            }
        }
        MPI_Group_incl(mpi_world_group, group_indices.size(), group_indices.data(), &level_group);
        // Build communicator
        if(level_group != MPI_GROUP_NULL) {
            MPI_Comm_create_group(MPI_COMM_WORLD, level_group, 0, &level_comm);
            MPI_Comm_split(level_comm, rank(level_comm) / base_, world_rank, &next_comm);
            comms_.push_back(next_comm);
            MPI_Comm_free(&level_comm);
            MPI_Group_free(&level_group);
        }
    }
    MPI_Group_free(&mpi_world_group);
}

// fix this at some point
Heirarchy::~Heirarchy() {
    // for(auto comm : comms_) {
    //     MPI_Comm_free(&comm);
    // }
}
}