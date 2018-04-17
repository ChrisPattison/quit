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

#pragma once
#include <vector>
#include <utility>
#include "types.hpp"
#include "graph.hpp"
#include "random_number_generator.hpp"
#include "log_lookup.hpp"

namespace propane {
class SpinVectorMonteCarlo {
protected:
    struct StateVector : std::vector<VertexType> {
        double gamma;
        double lambda;
        double beta;
    };

    util::LogLookup log_lookup_;
    RandomNumberGenerator rng_;

    Graph structure_;
/** Returns the overlap between replicas alpha and beta.
 */
    #pragma omp declare simd
    double Overlap(StateVector& alpha, StateVector& beta);
/**
 * Uses a look up table to compute a bound on the logarithm of a random number 
 * and compares to the exponent of the acceptance probability.
 * If the probability is inside the bound given by the look table, 
 * true exponential is computed and compared.
 */
    #pragma omp declare simd
    bool AcceptedMove(double log_probability);
/** Returns true if a move is accepted according to the Metropolis algorithm.
 */
    #pragma omp declare simd
    bool MetropolisAcceptedMove(double delta_energy, double beta);
/** Returns the energy of a replica
 * Implemented as the sum of elementwise multiplication of the replica vector with the 
 * product of matrix multiplication between the upper half of the adjacency matrix
 * and the replica.
 */
    #pragma omp declare simd
    double Hamiltonian(const StateVector& replica);
/** Projects replica onto classical spins
 */
    StateVector Project(const StateVector& replica); 
/** Returns the energy of the replica as given by the original problem Hamiltonian
 * Does not include strength prefactor
 */
    #pragma omp declare simd
    double ProblemHamiltonian(const StateVector& replica);
/** Returns energy of the replica as given by the Hamiltonian driving term
 * Does not include strength prefactor
 */
    #pragma omp declare simd
    double DriverHamiltonian(const StateVector& replica);
/** Returns the local field at site vertex
 */
    FieldType LocalField(StateVector& replica, IndexType vertex);
/** Returns the energy change associated with flipping spin vertex.
 * Implemented as the dot product of row vertex of the adjacency matrix 
 * with the replica vector multiplied by the spin at vertex.
 */
    double DeltaEnergy(StateVector& replica, IndexType vertex, FieldType new_value);
/** Carries out moves micro canonical sweeps
 */
    #pragma omp declare simd
    void MicroCanonicalSweep(StateVector& replica, std::size_t sweeps);
/** Carries out moves monte carlo sweeps of replica using the Metropolis algorithm.
 */
    #pragma omp declare simd
    void MetropolisSweep(StateVector& replica, std::size_t moves);
/** Carries out moves monte carlo sweeps of replica using the Heatbath algorithm.
 */
    #pragma omp declare simd
    void HeatbathSweep(StateVector& replica, std::size_t moves);
/** Sets the transverse field (gamma)
 */
    void TransverseField(StateVector& replica, double magnitude, double p_magnitude);
};
}
