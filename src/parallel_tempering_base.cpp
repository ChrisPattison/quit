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

#include "parallel_tempering_base.hpp"
#include <numeric>
#include <cassert>

namespace propane {
    auto ParallelTemperingBase::Bin::operator+=(const Bin& other) -> Bin {
        assert(gamma == other.gamma);
        assert(lambda == other.lambda);
        assert(beta == other.beta);

        samples += other.samples;
        exchange_probabilty += other.exchange_probabilty;
        average_energy += other.average_energy;
        ground_energy = std::min(ground_energy, other.ground_energy);

        problem_energy += other.problem_energy;
        driver_energy += other.driver_energy;
        return *this;
    }

    auto ParallelTemperingBase::Bin::operator+(const Bin& other) const -> Bin {
        Bin result = *this;
        result += other;
        return result;
    }

    auto ParallelTemperingBase::Bin::Finalize() -> Result {
        Result result;
        result.beta = beta;
        result.gamma = gamma;
        result.lambda = lambda;

        result.samples = samples;
        result.problem_energy = problem_energy / samples;
        result.driver_energy = driver_energy / samples;
        result.exchange_probabilty = exchange_probabilty / samples;
        result.average_energy = average_energy / samples;
        result.ground_energy = ground_energy;
        result.total_sweeps = total_sweeps;
        result.total_time = total_time;
        return result;
    }
}