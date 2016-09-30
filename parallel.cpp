#include "parallel.hpp"
#include <cmath>
#include <cassert>

Parallel::Parallel() {
    int initialized;
    MPI_Initialized(&initialized);
    if(!initialized) {
        MPI_Init(0,nullptr);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

    int finalized;
    MPI_Finalized(&finalized);
    if(finalized) {
        return;
    }

    tag_ = 0;
    MPI_Group mpi_world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &mpi_world_group);

    int level_stride = 1; // level_stride = kHeirarchyBase ^ level
    int max_levels = std::ceil(std::log(world_size_)/std::log(kHeirarchyBase));
    for(int level = 0; world_rank_ % level_stride == 0 && level < max_levels; ++level, level_stride *= kHeirarchyBase) {
        MPI_Comm level_comm;
        MPI_Comm next_comm;
        MPI_Group level_group;
        int kmax = world_size_ / level_stride;

        // Build group
        std::vector<int> group_indices;
        group_indices.reserve(kmax);
        for(int k = 0; k <= kmax; ++k) {
            int rank = k*level_stride;
            if(rank < world_size_) {
                group_indices.push_back(rank);
            }
        }
        MPI_Group_incl(mpi_world_group, group_indices.size(), group_indices.data(), &level_group);
        // Build communicator
        if(level_group != MPI_GROUP_NULL) {
            MPI_Comm_create_group(MPI_COMM_WORLD, level_group, 0, &level_comm);
            MPI_Comm_split(level_comm, rank(level_comm) / kHeirarchyBase, world_rank_, &next_comm);
            comm_heirarchy_.push_back(next_comm);
            MPI_Comm_free(&level_comm);
            MPI_Group_free(&level_group);
        }
    }
}

Parallel::~Parallel() {
    int finalized;
    MPI_Finalized(&finalized);
    if(!finalized) {
        MPI_Finalize();
    }
}

int Parallel::GetTag() {
    return ++tag_;
}

int Parallel::rank() {
    return world_rank_;
}

int Parallel::rank(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int Parallel::size() {
    return world_size_;
}

int Parallel::size(MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}

bool Parallel::is_root() {
    return rank() == kRoot;
}

bool Parallel::is_root(MPI_Comm comm) {
    return rank(comm) == kRoot;
}

void Parallel::ExecRoot(std::function<void()> target, MPI_Comm comm) {
    if(is_root(comm)) {
        target();
    }
}

void Parallel::Barrier(MPI_Comm comm) {
    MPI_Barrier(comm);
}

void Parallel::Barrier() {
    Barrier(MPI_COMM_WORLD);
}