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