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
 
// Call with graph 
// Call into object with spin vector vector
//
#include <vector>
#include <functional>
#include "graph.hpp"

namespace propane {
/** Driver for Hardware Monte Carlo accelerator
 */
class MonteCarloDriver {
    const int kVendorID = 0x7075;
    const int kDeviceID = 0x1337;

    const std::size_t kMemSize = 0x10'000'000;

    const std::size_t kSpinBase = 0x4000;
    const std::size_t kLutAddr = 0x3FC0;

    const std::size_t kDimensionAddr = 0x0000;
    const std::size_t kSpinAddr = 0x0004;
    const std::size_t kSweepAddr = 0x0008;
    const std::size_t kSeedAddr = 0x000c;
    const std::size_t kLutEntriesAddr = 0x0010;

    volatile std::uint8_t* bar0_;

    std::vector<double> local_field_;
    int bar0_disc_;

    int bus_;
    int device_;

    bool MapBar();
public:
    MonteCarloDriver();

    ~MonteCarloDriver();

    MonteCarloDriver& operator=(const MonteCarloDriver&) = delete;

    MonteCarloDriver& operator=(MonteCarloDriver&&);

    MonteCarloDriver(const MonteCarloDriver&) = delete;

    MonteCarloDriver(MonteCarloDriver&&);
/** Seed RNGs
 */
    void SeedRng(std::uint32_t seed);
/** Perform sweeps Monte Carlo sweeps on replica
 */
    void Sweep(std::vector<std::uint32_t>* replica, std::uint32_t sweeps);

/** Wait for completion of sweeps
 */
    void CompletionWait();

/** Set dE tables
 */
    void SetGraph(Graph& structure);
/** Set acceptance probabilities
 * This should not assume a Hamiltonian in the future
 * Assumes that indices match machine for now
 */
    void SetProb(double beta);
};
}