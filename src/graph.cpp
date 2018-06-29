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

void Graph::Resize(IndexType no_vertices) {
    fields_.resize(no_vertices);
    std::fill(fields_.begin(), fields_.end(), 0);
    adjacent_.clear();
    adjacent_.resize(no_vertices);

    weights_.clear();
    weights_.resize(no_vertices);

    fields_.shrink_to_fit();
    adjacent_.shrink_to_fit();
    weights_.shrink_to_fit();
}

void Graph::SetField(IndexType vertex, EdgeType field) {
    fields_.at(vertex) = field;
}

void Graph::AddEdge(IndexType from, IndexType to, EdgeType coupler) {
    adjacent_.at(from).push_back(to);
    weights_.at(from).push_back(coupler);
}

void Graph::Compress() {
    for(std::size_t k = 0; k < adjacent_.size(); ++k) {
        std::vector<std::pair<decltype(adjacent_)::value_type::value_type, decltype(weights_)::value_type::value_type>> key_value(adjacent_[k].size());
        for(std::size_t i = 0; i < adjacent_[k].size(); ++i) {
            key_value[i] = {adjacent_[k][i], weights_[k][i]};
        }
        std::sort(key_value.begin(), key_value.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for(std::size_t i = 0; i < adjacent_[k].size(); ++i) {
            adjacent_[k][i] = key_value[i].first;
            weights_[k][i] = key_value[i].second;
        }
    }
    for(auto v : adjacent_) { v.shrink_to_fit(); }
    for(auto v : weights_) { v.shrink_to_fit(); }
}
}