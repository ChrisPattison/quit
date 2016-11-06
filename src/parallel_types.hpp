#pragma once
#include <mpi.h>
#include <type_traits>
#include <vector>

namespace parallel
{
/** Container for the tree levels required in the heiarchial reductions.
 */
class Heirarchy {
    int levels_;
    int base_;
    std::vector<MPI_Comm> comms_;

    int rank(MPI_Comm comm);
public:
    Heirarchy();
    Heirarchy(const Heirarchy& source) = delete;
/** Builds a tree with FAN-IN base.
 */
    Heirarchy(int base);
/** Deletes frees communicators.
 */
    ~Heirarchy();
/** Const reference to the commuicator list.
 */
    const std::vector<MPI_Comm>& comms();
/** Total number of levels in the tree.
 */
    int global_levels();
/** Levels in the tree seen by current process.
 */
    int local_levels();
/** Gets FAN-IN of tree.
 */
    int base();
};

/** Container for data with a particular source or destination.
 */
template<typename T, typename = std::enable_if_t<std::is_trivially_copyable<T>::value, void>> struct Packet {
    int rank;
    std::vector<T> data;
};

/** Base class for asynchronous MPI operations.
 * The buffer is moved to buffer_ and the pointer passed to MPI.
 * On destruction the associated request is Wait'd so the buffer can be freed.
 */
template<typename T> class AsyncOp {
    friend class Mpi;
protected:
    MPI_Request request_;
    std::vector<T> buffer_;
public:
    AsyncOp();
    virtual ~AsyncOp();
/**
 * Copying would result result in the duplication of requests with only one instance actually holding the data.
 */
    AsyncOp(const AsyncOp&) = delete;
    AsyncOp(AsyncOp&& other);
/**
 * Copying would result result in the duplication of requests with only one instance actually holding the data.
 */
    AsyncOp& operator=(const AsyncOp&) = delete;
    AsyncOp& operator=(AsyncOp&& other);
/** Waits for completion of the aassociated request.
 */
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