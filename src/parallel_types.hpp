#pragma once
#include <mpi.h>
#include <type_traits>
#include <vector>

namespace parallel
{
class Heirarchy {
    int levels_;
    int base_;
    std::vector<MPI_Comm> comms_;

    int rank(MPI_Comm comm);
public:
    Heirarchy();
    Heirarchy(const Heirarchy& source) = delete;
    Heirarchy(int base);
    ~Heirarchy();

    const std::vector<MPI_Comm>& comms();
    int global_levels();
    int local_levels();
    int base();
};

template<typename T, typename = std::enable_if_t<std::is_trivially_copyable<T>::value, void>> struct Packet {
    int rank;
    std::vector<T> data;
};

template<typename T> class AsyncOp {
    friend class Mpi;
protected:
    MPI_Request request_;
    std::vector<T> buffer_;
public:
    AsyncOp();
    virtual ~AsyncOp();
    AsyncOp(const AsyncOp&) = delete;
    AsyncOp(AsyncOp&& other);
    AsyncOp& operator=(const AsyncOp&) = delete;
    AsyncOp& operator=(AsyncOp&& other);
    void Wait();
};

template<typename T> AsyncOp<T>::AsyncOp() {
    request_ = MPI_REQUEST_NULL;
}

template<typename T> AsyncOp<T>::AsyncOp(AsyncOp&& other) {
    buffer_.swap(other.buffer_);
    request_ = other.request_;
    other.request_ = MPI_REQUEST_NULL;
}

template<typename T> AsyncOp<T>::~AsyncOp() {
    Wait();
}

template<typename T> AsyncOp<T>& AsyncOp<T>::operator=(AsyncOp&& other) {
    buffer_.swap(other.buffer_);
    request_ = other.request_;
    other.request_ = MPI_REQUEST_NULL;
    return *this;
}

template<typename T> void AsyncOp<T>::Wait() {
    MPI_Wait(&request_, MPI_STATUS_IGNORE);
}
}