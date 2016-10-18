#include "parallel.hpp"
#include <cmath>
#include <cassert>

namespace parallel
{
Mpi::Mpi() {
    int initialized;
    MPI_Initialized(&initialized);
    if(!initialized) {
        MPI_Init(0,nullptr);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    world_rank_ = rank(MPI_COMM_WORLD);

    int finalized;
    MPI_Finalized(&finalized);
    if(finalized) {
        return;
    }

    tag_ = 0;
    vector_heirarchy = Heirarchy(kVectorHeirarchyBase);
    scalar_heirarchy = Heirarchy(kScalarHeirarchyBase);
}

Mpi::~Mpi() {
    int finalized;
    MPI_Finalized(&finalized);
    if(!finalized) {
        MPI_Finalize();
    }
}

int Mpi::GetTag() {
    ++tag_;
    return tag_;
}

void Mpi::IncrTag(int count) {
    assert(count >= 0 );
    tag_ += count;
}

int Mpi::rank() {
    return world_rank_;
}

int Mpi::rank(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int Mpi::size() {
    return world_size_;
}

int Mpi::size(MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}

bool Mpi::is_root() {
    return rank() == kRoot;
}

bool Mpi::is_root(MPI_Comm comm) {
    return rank(comm) == kRoot;
}

void Mpi::ExecRoot(std::function<void()> target, MPI_Comm comm) {
    if(is_root(comm)) {
        target();
    }
}

void Mpi::Barrier(MPI_Comm comm) {
    MPI_Barrier(comm);
}

void Mpi::Barrier() {
    Barrier(MPI_COMM_WORLD);
}

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