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