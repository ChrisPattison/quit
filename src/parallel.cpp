#include "parallel.hpp"
#include <cmath>
#include <cassert>

namespace parallel {
    
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
}