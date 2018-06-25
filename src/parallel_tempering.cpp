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
#include "compare.hpp"

namespace propane {

ParallelTempering::ParallelTempering(const Graph& structure, Config config) {
    if(config.seed != 0) {
        rng_ = RandomNumberGenerator(config.seed);
    }
    
    schedule_ = config.schedule;
    bin_set_ = config.bin_set;
    structure_ = structure;
    structure_.Adjacent().makeCompressed();
    solver_mode_ = config.solver_mode;
    sweeps_ = config.sweeps;
    planted_energy_ = config.planted_energy;
    hit_criteria_ = config.hit_criteria;

    field_.resize(structure_.Fields().size());
    for(int k = 0; k < structure_.Fields().size(); ++k) {
        // No transverse field
        field_[k] = FieldType(structure_.Fields()(k), 0.);
    }
    std::stable_sort(schedule_.begin(), schedule_.end(), [](const auto& left, const auto& right) {return left.gamma < right.gamma;});
    std::stable_sort(bin_set_.begin(), bin_set_.end());
}

std::vector<ParallelTempering::Result> ParallelTempering::Run() {
    std::vector<Bin> results;
    std::vector<Bin> result_sum; // One bin for each temperature
    replicas_.reserve(schedule_.size());

    // Initialize replicas
    for(auto temp : schedule_) {
        replicas_.emplace_back();
        auto& replica = replicas_.back();
        replica.beta = temp.beta;
        replica.gamma = temp.gamma;
        replica.lambda = temp.lambda;
        replica.resize(structure_.size());
        for(std::size_t k = 0; k < replica.size(); ++k) {
            replica[k] = FieldType(rng_.Probability());
        }
    }
    
    // Initialize result_sum
    result_sum.resize(schedule_.size());
    for(int i = 0; i < result_sum.size(); ++i) {
        result_sum[i].gamma = schedule_[i].gamma;
        result_sum[i].lambda = schedule_[i].lambda;
        result_sum[i].beta = schedule_[i].beta;
    }

    bool groundstate_found = false;
    // Run
    auto total_time_start = std::chrono::high_resolution_clock::now();
    for(std::size_t count = 0; (count < sweeps_) && !groundstate_found; ++count) {
        // Sweep replicas
        for(int k = 0; k < schedule_.size(); ++k) {
            MicroCanonicalSweep(replicas_[k], schedule_[k].microcanonical);
            MetropolisSweep(replicas_[k], schedule_[k].metropolis);
            HeatbathSweep(replicas_[k], schedule_[k].heatbath);
        }
        // Do replica exchange
        // This could reuse the projected energy computed for observables
        auto exchange_probabilty = ReplicaExchange(replicas_);

        // Measure observables
        for(int i = 0; i < result_sum.size(); ++i) {
            auto step_observables = Observables(replicas_[i]);
            step_observables.exchange_probabilty = exchange_probabilty[i];
            result_sum[i] += step_observables;
        }

        double min_energy = std::min_element(result_sum.begin(), result_sum.end(), [](const auto& a, const auto& b) { return a.ground_energy < b.ground_energy; })->ground_energy;
        groundstate_found = min_energy <= planted_energy_|| util::FuzzyEpsCompare(min_energy, planted_energy_, hit_criteria_);

        if(groundstate_found || std::binary_search(bin_set_.begin(), bin_set_.end(), count+1)) { // Record Bin
            auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - total_time_start).count();
            auto new_results = results.insert(results.end(), result_sum.begin(), result_sum.end());
            std::transform(new_results, results.end(), new_results, [&](Bin b) { 
                b.total_time = total_time;
                b.total_sweeps = count+1;
                return b;
            });
        }
    }

    std::vector<Result> final_results;
    final_results.resize(results.size());
    std::transform(results.begin(), results.end(), final_results.begin(), [](auto& r) { return r.Finalize(); });
    return final_results;
}

std::vector<double> ParallelTempering::ReplicaExchange(std::vector<StateVector>& replica_set) {
    std::vector<double> problem_energy(replica_set.size());
    std::vector<double> driver_energy(replica_set.size());
    std::vector<double> exchange_probabilty(replica_set.size());
    std::transform(replica_set.begin(), replica_set.end(), problem_energy.begin(), [&](StateVector& r) { return ProblemHamiltonian(r); });
    std::transform(replica_set.begin(), replica_set.end(), driver_energy.begin(), [&](StateVector& r) { return DriverHamiltonian(r); });
    // replica_set[k+1].gamma > replica_set[k].gamma should be true
    for(int k = 0; k < schedule_.size()-1; ++k) {
        // Exp[-d[AB] H_P - d[DB] H_D
        exchange_probabilty[k] = std::min(1.0,
            std::exp(
                (replica_set[k+1].gamma*replica_set[k+1].beta - replica_set[k].gamma*replica_set[k].beta)
                *(driver_energy[k+1] - driver_energy[k])
                +(replica_set[k+1].lambda*replica_set[k+1].beta - replica_set[k].lambda*replica_set[k].beta)
                *(problem_energy[k+1] - problem_energy[k])));
        if(exchange_probabilty[k] > rng_.Probability()) {
            std::swap(replica_set[k].beta, replica_set[k+1].beta);
            std::swap(replica_set[k].gamma, replica_set[k+1].gamma);
            std::swap(replica_set[k].lambda, replica_set[k+1].lambda);
            std::swap(replica_set[k], replica_set[k+1]);
            std::swap(problem_energy[k], problem_energy[k+1]);
        }
    }
    exchange_probabilty.back() = std::numeric_limits<double>::quiet_NaN();
    return exchange_probabilty;
}

auto ParallelTempering::Observables(const StateVector& replica, bool minimum_set) -> Bin {
    Bin result;
    result.gamma = replica.gamma;
    result.lambda = replica.lambda;
    result.beta = replica.beta;
    result.samples = 1;

    if(!minimum_set) {
        result.problem_energy = ProblemHamiltonian(replica);
        result.driver_energy = DriverHamiltonian(replica);
    }

    auto projected_energy = ProblemHamiltonian(Project(replica));
    result.discrete_energy = projected_energy;
    result.ground_energy = projected_energy;
    return result;
}
}
