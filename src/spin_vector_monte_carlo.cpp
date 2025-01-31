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
    // Priviliged direction
    auto direction = std::accumulate(replica.begin(), replica.end(), FieldType(0,0,0));
    direction /= direction * direction;
    // We should check both directions but randomly flip for now
    direction *= rng_.Probability() < 0.5 ? -1.0 : 1.0;

    StateVector projected;
    projected.resize(replica.size());
    for( std::size_t k = 0; k < replica.size(); ++k ) {
        projected[k] = replica[k] * direction > 0 ? FieldType(1.0, 0.0, 0.0) : FieldType(-1.0, 0.0, 0.0);
    }
    return projected;
}

double SpinVectorMonteCarlo::ProblemHamiltonian(const StateVector& replica) {
    double energy = 0.0;
    for(auto site = 0; site < structure_.size(); ++site) {
        energy += ((replica[site] / 2.0)
            * std::inner_product( 
            structure_.adjacent()[site].begin(), structure_.adjacent()[site].end(),
            structure_.weights()[site].begin(),
            FieldType(0,0,0), std::plus<>(), 
            [&replica](const auto& spin, const auto& weight) { return weight * replica[spin]; }));
    }
    energy += std::inner_product(
        structure_.fields().begin(), structure_.fields().end(),
        replica.begin(),
        0.0, std::plus<>(),
        [&replica](const auto& field, const auto& spin) { return field * spin[0]; });

    return energy;
}

double SpinVectorMonteCarlo::DriverHamiltonian(const StateVector& replica) {
    double energy = 0.0;
    energy += replica.gamma * 
        std::accumulate(
        replica.begin(), replica.end(),
        0.0, [&replica](const auto& a, const auto& spin) { return a + spin[1]; });
    return energy;
}

FieldType SpinVectorMonteCarlo::LocalField(StateVector& replica, IndexType vertex) {
    FieldType h(0, 0);
    const auto adj_count = structure_.adjacent()[vertex].size();
    for(std::size_t i = 0; i < adj_count; ++i) {
        h += replica.lambda * structure_.weights()[vertex][i] * replica[structure_.adjacent()[vertex][i]];
    }
    h[0] += replica.lambda * structure_.fields()[vertex];
    h[1] += replica.gamma;
    return h;
}

double SpinVectorMonteCarlo::DeltaEnergy(StateVector& replica, IndexType vertex, FieldType new_value) {
    return (new_value - replica[vertex]) * LocalField(replica, vertex);
}

void SpinVectorMonteCarlo::MicroCanonicalSweep(std::vector<StateVector>& replica, std::size_t sweeps) {
    std::vector<StateVector> local_field;
    local_field.resize(replica.size());
    for(std::size_t k = 0; k < replica.size(); ++k) {
        local_field[k].resize(replica[k].size());
        for(std::size_t i = 0; i < replica[k].size(); ++i) {
            local_field[k][i] = LocalField(replica[k], i);
        }
    }

    auto var_count = replica[0].size();

    for(std::size_t s = 0; s < sweeps; ++s) {
        for(IndexType v = 0; v < replica[0].size(); ++v) {
            auto vertex = rng_.Range(var_count);
            const auto& adjacent = structure_.adjacent()[vertex];
            const auto& weight = structure_.weights()[vertex];

            for(std::size_t k = 0; k < replica.size(); ++k) {
                auto& r = replica[k];
                // get local field
                auto& h = local_field[k][vertex];
                auto new_spin = ((2*h*(r[vertex]*h))/(h*h))-r[vertex];
                auto value_diff = new_spin - r[vertex];
                r[vertex] = new_spin;

                for(std::size_t i = 0; i < adjacent.size(); ++i) {
                    local_field[k][adjacent[i]][0] += r.lambda * (weight[i] * value_diff)[0];
                }
            }
        }
    }
}

void SpinVectorMonteCarlo::MetropolisSweep(std::vector<StateVector>& replica, std::size_t sweeps) {
    std::vector<StateVector> local_field;
    local_field.resize(replica.size());
    for(std::size_t k = 0; k < replica.size(); ++k) {
        local_field[k].resize(replica[k].size());
        for(std::size_t i = 0; i < replica[k].size(); ++i) {
            local_field[k][i] = LocalField(replica[k], i);
        }
    }

    auto var_count = replica[0].size();

    for(std::size_t s = 0; s < sweeps; ++s) {
        for(IndexType v = 0; v < var_count; ++v) {
            IndexType vertex = rng_.Range(var_count);

            for(std::size_t k = 0; k < replica.size(); ++k) {
                auto& r = replica[k];
                auto raw_new_value1 = sin_lookup_.Unit(rng_.Probability());
                auto raw_new_value2 = sin_lookup_.Unit(rng_.Probability());
                auto new_value = VertexType(
                    raw_new_value1.sin * raw_new_value2.cos, 
                    raw_new_value1.sin * raw_new_value2.sin,
                    raw_new_value1.cos);

                auto value_diff = new_value - r[vertex];
                double delta_energy = value_diff * local_field[k][vertex];
                
                //round-off isn't a concern here
                if(MetropolisAcceptedMove(delta_energy, r.beta)) {
                    r[vertex] = new_value;
                    const auto& adjacent = structure_.adjacent()[vertex];
                    const auto& weight = structure_.weights()[vertex];
                    // Update local fields
                    for(std::size_t i = 0; i < adjacent.size(); ++i) {
                        local_field[k][adjacent[i]] += r.lambda * (weight[i] * value_diff);
                    }
                }
            }
        }
    }
}

void SpinVectorMonteCarlo::HeatbathSweep(std::vector<StateVector>& replica, std::size_t sweeps) {
    // auto var_count = replica[0].size();
    
    // for(std::size_t s = 0; s < sweeps; ++s) {
    //     for(IndexType i = 0; i < var_count; ++i) {
    //         IndexType vertex = rng_.Range(var_count);

    //         for(std::size_t k = 0; k < replica.size(); ++k) {
    //             auto& r = replica[k];
    //             auto h = LocalField(r, vertex);
    //             auto h_mag = std::sqrt(h*h);
    //             if (h_mag != 0) {
    //                 auto h_unit = h / h_mag;
    //                 float prob = rng_.Probability();
    //                 double x = -std::log(1 + prob * (std::exp(-2 * r.beta * h_mag) - 1)) / (r.beta * h_mag) - 1.;
    //                 auto h_perp = FieldType(h_unit[1], -h_unit[0]);
    //                 prob = rng_.Probability();
    //                 h_perp *= prob < 0.5 ? -1 : 1;
    //                 r[vertex] = h_unit * x + h_perp * std::sqrt(1.0-x*x);
    //             }else {
    //                 r[vertex] = VertexType(rng_.Probability());
    //             }
    //         }
    //     }
    // }
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

