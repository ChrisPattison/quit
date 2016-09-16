#include "parallel.hpp"

Parallel::Parallel() {
    MPI_Init(0,nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
}

int Parallel::rank() {
    return rank_;
}

int Parallel::size() {
    return size_;
}

bool Parallel::is_root() {
    return rank() == kRoot;
}

void Parallel::ExecRoot(std::function<void()> target) {
    if(rank_ == 0) {
        target();
    }
}