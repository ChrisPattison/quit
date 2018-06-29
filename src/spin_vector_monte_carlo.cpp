/* Copyright (c) 2017 C. Pattison
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

#include "spin_vector_monte_carlo.hpp"
#include "compare.hpp"
#include <numeric>
#include <algorithm>

namespace propane {

double SpinVectorMonteCarlo::Hamiltonian(const StateVector& replica) {
    return replica.lambda * ProblemHamiltonian(replica) + replica.gamma * DriverHamiltonian(replica);
}

SpinVectorMonteCarlo::StateVector SpinVectorMonteCarlo::Project(const StateVector& replica) {
    StateVector projected;
    projected.resize(replica.size());
    for( std::size_t k = 0; k < replica.size(); ++k ) {
        projected[k] = replica[k] * FieldType(1.0, 0.0) > 0 ? FieldType(1.0, 0.0) : FieldType(-1.0, 0.0);
    }
    return projected;
}

double SpinVectorMonteCarlo::ProblemHamiltonian(const StateVector& replica) {
    double energy = 0.0;
    for(auto site = 0; site < structure_.size(); ++site) {
        energy += replica[site][0] / 2.0
            * std::inner_product( 
            structure_.adjacent()[site].begin(), structure_.adjacent()[site].end(),
            structure_.weights()[site].begin(),
            0.0, std::plus<>(), 
            [&replica](const auto& spin, const auto& weight) { return weight * replica[spin][0]; });
    }
    energy -= std::inner_product(
        structure_.fields().begin(), structure_.fields().end(),
        replica.begin(),
        0.0, std::plus<>(),
        [&replica](const auto& field, const auto& spin) { return field * spin[0]; });

    return energy;
}

double SpinVectorMonteCarlo::DriverHamiltonian(const StateVector& replica) {
    double energy = 0.0;
    energy -= replica.gamma * 
        std::accumulate(
        replica.begin(), replica.end(),
        0.0, [&replica](const auto& a, const auto& spin) { return a + spin[1]; });
    return energy;
}

FieldType SpinVectorMonteCarlo::LocalField(StateVector& replica, IndexType vertex) {
    FieldType h(0, 0);
    h[0] += replica.lambda * std::inner_product(
        structure_.adjacent()[vertex].begin(), structure_.adjacent()[vertex].end(),
        structure_.weights()[vertex].begin(),
        0.0, std::plus<>(), 
        [&replica](const auto& spin, const auto& weight) { return weight * replica[spin][0]; });

    h[0] -= replica.lambda * structure_.fields()[vertex];
    h[1] -= replica.gamma;
    return h;
}

double SpinVectorMonteCarlo::DeltaEnergy(StateVector& replica, IndexType vertex, FieldType new_value) {
    return (new_value - replica[vertex]) * LocalField(replica, vertex);
}

void SpinVectorMonteCarlo::MicroCanonicalSweep(StateVector& replica, std::size_t sweeps) {
    for(std::size_t k = 0; k < sweeps; ++k) {
        for(IndexType i = 0; i < replica.size(); ++i) {
            auto vertex = i;
            
            // get local field
            auto h = LocalField(replica, vertex);
            auto new_spin = ((2*h*(replica[vertex]*h))/(h*h))-replica[vertex];
            replica[vertex] = new_spin;
        }
    }
}

void SpinVectorMonteCarlo::MetropolisSweep(StateVector& replica, std::size_t sweeps) {
    StateVector local_field;
    local_field.resize(replica.size());
    for(std::size_t i = 0; i < replica.size(); ++i) {
        local_field[i] = LocalField(replica, i);
    }

    for(std::size_t k = 0; k < sweeps; ++k) {
        for(IndexType i = 0; i < replica.size(); ++i) {
            IndexType vertex = rng_.Range(replica.size());
            auto raw_new_value = sin_lookup_.Unit(rng_.Probability());
            auto new_value = VertexType(raw_new_value.cos, raw_new_value.sin);

            auto value_diff = new_value - replica[vertex];
            double delta_energy = value_diff * local_field[i];
            
            //round-off isn't a concern here
            if(MetropolisAcceptedMove(delta_energy, replica.beta)) {
                replica[vertex] = new_value;
                const auto& adjacent = structure_.adjacent()[vertex];
                const auto& weight = structure_.weights()[vertex];
                local_field[vertex][1] += replica.gamma * value_diff[1];
                local_field[vertex][0] += replica.lambda * structure_.fields()[vertex] * value_diff[0];
                // Update local fields
                for(std::size_t i = 0; i < adjacent.size(); ++i) {
                    local_field[adjacent[i]][0] += replica.lambda * (weight[i] * value_diff)[0];
                }
            }
        }
    }
}

void SpinVectorMonteCarlo::HeatbathSweep(StateVector& replica, std::size_t sweeps) {
    for(std::size_t k = 0; k < sweeps; ++k) {
        for(IndexType i = 0; i < replica.size(); ++i) {
            IndexType vertex = rng_.Range(replica.size());
            auto h = LocalField(replica, vertex);
            auto h_mag = std::sqrt(h*h);
            if (h_mag != 0) {
                auto h_unit = h / h_mag;
                float prob = rng_.Probability();
                double x = -std::log(1 + prob * (std::exp(-2 * replica.beta * h_mag) - 1)) / (replica.beta * h_mag) - 1.;
                auto h_perp = FieldType(h_unit[1], -h_unit[0]);
                prob = rng_.Probability();
                h_perp *= prob < 0.5 ? -1 : 1;
                replica[vertex] = h_unit * x + h_perp * std::sqrt(1.0-x*x);
            }else {
                replica[vertex] = VertexType(rng_.Probability());
            }
        }
    }
}

double SpinVectorMonteCarlo::Overlap(StateVector& alpha, StateVector& beta) {
    return std::inner_product(alpha.begin(), alpha.end(), beta.begin(), 0.0) / structure_.size();
}

bool SpinVectorMonteCarlo::MetropolisAcceptedMove(double delta_energy, double beta) {
    if(delta_energy < 0.0) {
        return true;
    }
    
    double acceptance_prob_exp = -delta_energy*beta;
    return AcceptedMove(acceptance_prob_exp);
}

bool SpinVectorMonteCarlo::AcceptedMove(double log_probability) {
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

void SpinVectorMonteCarlo::TransverseField(StateVector& replica, double magnitude, double p_magnitude) {
    replica.gamma = magnitude;
}
}