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
#include <vector>
#include <utility>
#include "types.hpp"
#include "graph.hpp"
#include "types.hpp"
#include "random_number_generator.hpp"
#include "population_annealing.hpp"
#include "parallel.hpp"
#include <Eigen/Dense>

namespace propane {
/** Parallel version of PopulationAnnealing using MPI.
 */
class ParallelPopulationAnnealing : protected PopulationAnnealing {
protected:
/** Minimum exceeded fraction of average node population before redistribution will take place.
  */
    double kMaxPopulation = 1.10;

    parallel::Mpi parallel_;

    int average_node_population_;
/** Resamples population according to the Boltzmann distribution.
 * Attempts to maintain approximately the same population as detailed in arXiv:1508.05647
 * Returns the normalization factor Q as a byproduct.
 */
    double Resample(double new_beta, double population_fraction);
/** Redistributes population evenly among all processes.
 * Ordering of the replicas must remain preserved at all times.
 * Packs replicas in Packets to send to other processes.
 */
    void Redistribute();
/** Combines source with target additively.
 */
    void CombineHistogram(std::vector<Result::Histogram>& target, const std::vector<Result::Histogram>& source);
/** Vector of the integer population of all surviving families originating from current process.
 * Uses Packets to send the count of families to the originator who compiles full count.
 * Invalid if families are store non-contiguously.
 */
    std::vector<double> FamilyCount();
/** Packets replicas into a single vector
 */
    std::vector<VertexType> Pack(const std::vector<StateVector>& source);
/** Packets replicas into a single vector
 */
    std::vector<VertexType> Pack(const std::vector<StateVector>::const_iterator begin_iterator, const std::vector<StateVector>::const_iterator end_iterator);
/** Unpacks a single vector into a vector of replicas
 */
    std::vector<StateVector> Unpack(std::vector<VertexType>& source);
public:
/** Data relavent only to the parallel implementation
 */
    struct Result : PopulationAnnealing::Result {
        double redist_walltime = std::numeric_limits<double>::quiet_NaN();
        double observables_walltime = std::numeric_limits<double>::quiet_NaN();
        int max_family_size = -1;
    };
/** Config parameters relavent to only the parallel implementation
 */
    struct Config : PopulationAnnealing::Config {
        double max_population = 1.10;
    };
    ParallelPopulationAnnealing(const ParallelPopulationAnnealing&) = delete;
/** Intializes solver.
 * schedule specifies the annealing schedule, sweep counts, and histogram generation at each step.
 * seed may be zero in which case one will be generated.
 */
    ParallelPopulationAnnealing(Graph& structure, Config config);
/** Run solver and return results.
 */
    std::vector<Result> Run();
};
}