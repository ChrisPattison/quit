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
#include <limits>
#include <cstdint>

namespace propane {
/** Currently outputs already binned samples
 *  In the future, binning should be done in post processing to unify the PA and PT data output paths
 */
class ParallelTemperingBase {
public:
    struct Result;

    struct Bin {
        std::size_t samples = 0;
        double beta = std::numeric_limits<double>::quiet_NaN();
        
        double exchange_probabilty = 0.0;
        double energy = 0.0;
        double ground_energy = std::numeric_limits<double>::max();

        unsigned long long int total_sweeps = 0;
        double total_time = std::numeric_limits<double>::quiet_NaN();
/** Combine two sample sets
 */
        Bin operator+(const Bin& other) const;
        Bin operator+=(const Bin& other);

        Result Finalize();
    };

    struct Result : public Bin { };

    struct Schedule {
        double beta;
        bool compute_observables = false;
    };

    struct Config {
        std::size_t sweeps;
        std::uint64_t seed;
        std::vector<ParallelTemperingBase::Schedule> schedule;
        std::vector<std::size_t> bin_set;
        bool solver_mode = false;
        double planted_energy = std::numeric_limits<double>::quiet_NaN();
        double hit_criteria = 1e-12;
    };
};
}