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
 
#include "parallel_tempering.hpp"

namespace propane {

ParallelTempering::ParallelTempering(Graph& structure, Config config) {
    if(config.seed != 0) {
        rng_ = RandomNumberGenerator(config.seed);
    }
    
    schedule_ = config.schedule;
    structure_ = structure;
    structure_.Adjacent().makeCompressed();
    solver_mode_ = config.solver_mode;
    uniform_init_ = config.uniform_init;
    sweeps_ = config.sweeps;

    field_.resize(structure_.Fields().size());
    for(int k = 0; k < structure_.Fields().size(); ++k) {
        // No transverse field
        field_[k] = FieldType(structure_.Fields()(k), 0.);
    }
}

std::vector<ParallelTempering::Result> ParallelTempering::Run() {
    std::vector<Result> results;
    std::vector<Bin> bins;
    Bin result_sum;
    replicas_.reserve(schedule_.size());

    // Initialize
    for(auto temp : schedule_) {
        replicas_.emplace_back();
        auto& replica = replicas_.back();
        replica.beta = temp.beta;
        replica.gamma = temp.gamma;
        replica.resize(structure_.size());
        for(std::size_t k = 0; k < replica.size(); ++k) {
            if(uniform_init_) {
                replica[k] = FieldType(0.,1.);
            }else {
                replica[k] = FieldType(1.,0.) * (rng_.Probability() < 0.5 ? 1 : -1);
            }
        }
    }

    // Initialize result_sum
    result_sum.ground_energy = std::numeric_limits<decltype(result_sum.ground_energy)>::max();
    // Run
    auto total_time_start = std::chrono::high_resolution_clock::now();
    for(std::size_t count = 0; count < sweeps_; ++count) {
        // Do replica exchange
        ReplicaExchange(replicas_);
        // Sweep replicas
        for(int k = 0; k < schedule_.size(); ++k) {
            MicroCanonicalSweep(replicas_[k], schedule_[k].microcanonical);
            MetropolisSweep(replicas_[k], schedule_[k].metropolis);
        }

        // Measure observables.
        // Ground state only for now
        result_sum.ground_energy = std::min(result_sum.ground_energy, ProjectedHamiltonian(replicas_.front()));
    }

    results.emplace_back();
    results.back().ground_energy = result_sum.ground_energy;
    results.back().total_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - total_time_start).count();
    return results;
}

void ParallelTempering::ReplicaExchange(std::vector<StateVector>& replica_set) {
    std::vector<double> projected_energy;
    projected_energy.resize(replica_set.size());
    std::transform(replica_set.begin(), replica_set.end(), projected_energy.begin(), [&](StateVector& r) { return ProjectedHamiltonian(r); });
    // replica_set[k+1].gamma > replica_set[k].gamma should be true
    for(int k = 0; k < schedule_.size()-1; ++k) {
        double exchange_probabilty = std::min(1.0,
            std::exp((1./replica_set[k+1].gamma - 1./replica_set[k].gamma)*(projected_energy[k+1] - projected_energy[k])));
        if(exchange_probabilty < rng_.Probability()) {
            std::swap(replica_set[k], replica_set[k+1]);
            std::swap(projected_energy[k], projected_energy[k+1]);
        }
    }
}
}