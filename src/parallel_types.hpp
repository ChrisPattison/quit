/* Copyright (c) 2016 C. Pattison
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
#pragma once
#include <mpi.h>
#include <type_traits>
#include <vector>

namespace parallel {
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