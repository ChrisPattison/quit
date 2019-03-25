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

#pragma once
#include <vector>
#include <utility>
#include "types.hpp"
#include "graph.hpp"
#include "random_number_generator.hpp"
#include "log_lookup.hpp"

namespace propane {
class Ising {
protected:
    /** TODO: The encapsulation here is very broken
     */
    struct StateVector {
        double beta;
        double energy;
        std::vector<VertexType> state;
        std::vector<double> local_field;
        auto size() const { return state.size(); }
    };

    util::LogLookup log_lookup_;
    RandomNumberGenerator rng_;

    Graph structure_;
/**
 * Uses a look up table to compute a bound on the logarithm of a random number 
 * and compares to the exponent of the acceptance probability.
 * If the probability is inside the bound given by the look table, 
 * true exponential is computed and compared.
 */
    bool AcceptedMove(double log_probability);
/** Returns true if a move is accepted according to the Metropolis algorithm.
 */
    bool MetropolisAcceptedMove(double delta_energy, double beta);
/** Returns the local field at site
 */
    double LocalField(const StateVector& replica, IndexType site);
/** Initialize local fields
 */
    void InitLocalFields(StateVector* replica);
/** Returns the energy of a replica
 * Implemented as the sum of elementwise multiplication of the replica vector with the 
 * product of matrix multiplication between the upper half of the adjacency matrix
 * and the replica.
 */
    double Hamiltonian(const StateVector& replica);
/** Returns the energy change associated with flipping spin site.
 */
    double DeltaEnergy(const StateVector& replica, IndexType site);
/** Carries out moves monte carlo sweeps of replica using the Metropolis algorithm.
 */
    void MetropolisSweep(std::vector<StateVector>* replica, std::size_t moves);
};
}
