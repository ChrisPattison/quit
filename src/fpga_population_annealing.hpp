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
#include "population_annealing.hpp"
#include "types.hpp"
#include "graph.hpp"
#include "monte_carlo_driver.hpp"
#include <Eigen/Dense>

namespace propane {
/** Accelerated Implementation of Population Annealing Monte Carlo.
 */
class FpgaPopulationAnnealing : public PopulationAnnealing{
    MonteCarloDriver driver_;
public:
protected:

using SpinPack = std::array<std::uint64_t, 4>;
using StateVectorPack = Eigen::Matrix<SpinPack, Eigen::Dynamic, 1>;

/** Carries out moves monte carlo sweeps of all replicas on the accelerator.
 */
    void Sweep(int moves);
/** Returns true if a move may be made that reduces the total energy.
 */
    double Resample(double new_beta, double population_fraction);
/** Returns new population size
 * Uses a logistic curve with parameters given in input file
 * This probably will be removed in the future with preference given 
 * to the current method of specifying beta schedules (one per temperature)
 */
public:

    FpgaPopulationAnnealing() = delete;
/** Intializes solver.
 * The inputs will be replaced by a struct in the future.
 * schedule specifies the annealing schedule, sweep counts, and histogram generation at each step.
 * seed may be zero in which case one will be generated.
 */
    FpgaPopulationAnnealing(Graph& structure, Config schedule);
/** Run solver and return results.
 */
};
}