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
 
#include "graph.hpp"

namespace propane {

void Graph::Resize(int no_vertices, int no_couplers) {
    fields_.resize(no_vertices);
    fields_.setZero();
    field_nonzero_ = false;
    adjacent_.resize(no_vertices, no_vertices);
    adjacent_.reserve(no_couplers);
}

void Graph::SetField(int vertex, EdgeType field) {
    field_nonzero_ = true;
    fields_(vertex) = field;
}

void Graph::AddEdge(int from, int to, EdgeType coupler) {
    adjacent_.insert(from, to) = coupler;
}


bool Graph::IsConsistent() const {
    bool not_consistent = false;
    for(std::size_t k = 0; k < adjacent_.outerSize(); ++k) {
        bool zero = true;
        for(Eigen::SparseMatrix<EdgeType>::InnerIterator it(adjacent_, k); it; ++it) {
            if(std::abs(it.value()) > kEpsilon) {
                zero = false;
                break;
            }
        }
        if(zero) {
            not_consistent = true;
        }
    }
    return !(not_consistent | !adjacent_.isApprox(adjacent_.transpose()));
}

Eigen::SparseMatrix<EdgeType>& Graph::Adjacent() {
    return adjacent_;
}

Eigen::Matrix<EdgeType, Eigen::Dynamic, 1>& Graph::Fields() {
    return fields_;
}

bool Graph::has_field() const {
    return field_nonzero_;
}

int Graph::size() const {
    return fields_.size();
}

int Graph::edges() const {
    return adjacent_.nonZeros() / 2;
}
}