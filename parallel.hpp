#pragma once
#include <functional>
#include <vector>
#include <type_traits>
#include <mpi.h>

class Parallel {
    const int kRoot = 0;
    int rank_;
    int size_;
    
public:
    Parallel();

    Parallel(const Parallel&) = delete;

    int rank();

    int size();
    
    bool is_root();
    // Executes something on root rank
    void ExecRoot(std::function<void()> target);
    //TODO:: assert that T is a trivial type
    template<typename T> T Reduce(T value, std::function<T(std::vector<T>&)> reduce);
    template<typename T> T ReduceToAll(T value, std::function<T(std::vector<T>&)> reduce);
    template<typename T> T ReduceRootToAll(T value, std::function<T(std::vector<T>&)> reduce);
};

template<typename T> T Parallel::Reduce(T value, std::function<T(std::vector<T>&)> reduce) {
    std::vector<T> data_buffer;
    if(is_root()) {
        data_buffer.resize(size());
    }

    MPI_Gather(&value, sizeof(T), MPI_BYTE, data_buffer.data(), sizeof(T), MPI_BYTE, kRoot, MPI_COMM_WORLD);

    if(is_root()) {
        return reduce(data_buffer);
    }else {
        return {};
    }
}

template<typename T> T Parallel::ReduceRootToAll(T value, std::function<T(std::vector<T>&)> reduce) {
    T reducedvalue = Reduce(value, reduce);

    MPI_BCAST(&reducedvalue, sizeof(T), MPI_BYTE, kRoot, MPI_COMM_WORLD);
    return reducedvalue;
}

template<typename T> T Parallel::ReduceToAll(T value, std::function<T(std::vector<T>&)> reduce) {
    std::vector<T> data_buffer(sizeof(T) * size());

    MPI_Allgather(&value, sizeof(T), MPI_BYTE, data_buffer.data(), sizeof(T), MPI_BYTE, MPI_COMM_WORLD);

    return reduce(data_buffer);
}