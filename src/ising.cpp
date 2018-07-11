/* Copyright (c) 2018 C. Pattison
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

#include "ising.hpp"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath>

namespace propane {
bool Ising::AcceptedMove(double log_probability) {
    double test = rng_.Probability();
    // Compute bound on log of test number
    auto bound = log_lookup_(test);

    if(bound.upper < log_probability) {
        return true;
    }else if(bound.lower > log_probability) {
        return false;
    }
    // Compute exp if LUT can't resolve it
    return std::exp(log_probability) > test;
}

bool Ising::MetropolisAcceptedMove(double delta_energy, double beta) {
    if(delta_energy < 0.0) {
        return true;
    }
    
    double acceptance_prob_exp = -delta_energy*beta;
    return AcceptedMove(acceptance_prob_exp);
}

double Ising::LocalField(const StateVector& replica, IndexType site) {
    double field = 0;
    field += std::inner_product( 
        structure_.adjacent()[site].begin(), structure_.adjacent()[site].end(),
        structure_.weights()[site].begin(),
        0.0, std::plus<>(), 
        [&replica](const auto& spin, const auto& weight) { return weight * replica[spin]; });
    field += structure_.fields()[site];
    return field;
}

double Ising::Hamiltonian(const StateVector& replica) {
    double energy = 0.0;
    for(auto site = 0; site < structure_.size(); ++site) {
        energy += replica[site] / 2.0
            * std::inner_product( 
            structure_.adjacent()[site].begin(), structure_.adjacent()[site].end(),
            structure_.weights()[site].begin(),
            0.0, std::plus<>(), 
            [&replica](const auto& spin, const auto& weight) { return weight * replica[spin]; });
    }
    energy += std::inner_product(structure_.fields().begin(), structure_.fields().end(), replica.begin(), 0.0);

    return energy;
}

double Ising::DeltaEnergy(const StateVector& replica, IndexType site) {
    return -2*replica[site]*LocalField(replica, site);
}

void Ising::MetropolisSweep(std::vector<StateVector>* replica_set_ptr, std::size_t sweeps) {
    auto& replica_set = *replica_set_ptr;
    std::vector<StateVector> local_field;
    local_field.resize(replica_set.size());
    for(std::size_t k = 0; k < replica_set.size(); ++k) {
        local_field[k].resize(replica_set[k].size());
        for(std::size_t i = 0; i < replica_set[k].size(); ++i) {
            local_field[k][i] = LocalField(replica_set[k], i);
        }
    }

    auto var_count = replica_set[0].size();

    for(std::size_t s = 0; s < sweeps; ++s) {
        for(IndexType v = 0; v < var_count; ++v) {
            IndexType vertex = rng_.Range(var_count);

            for(std::size_t k = 0; k < replica_set.size(); ++k) {
                auto& r = replica_set[k];

                double delta_energy = DeltaEnergy(r, vertex);
                
                //round-off isn't a concern here
                if(MetropolisAcceptedMove(delta_energy, r.beta)) {
                    r[vertex] *= -1;
                    const auto& adjacent = structure_.adjacent()[vertex];
                    const auto& weight = structure_.weights()[vertex];
                    // Update local fields
                    for(std::size_t i = 0; i < adjacent.size(); ++i) {
                        local_field[k][adjacent[i]] += 2 * r[vertex] * weight[i];
                    }
                }
            }
        }
    }
}
}