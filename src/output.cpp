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
 
#include "output.hpp"
#include <iomanip>
#include <iostream>
#include <cfenv>
#include <string>
#include "compare.hpp"

namespace propane { namespace io {   
void Header(Graph& model, std::string config_path, std::string bond_path) {
    std::cout << "# Parallel Tempering" << std::endl;
    std::cout << "# C. Pattison" << std::endl;
    std::cout << "# Branch: " << version::kRefSpec << std::endl;
    std::cout << "# Commit: " << version::kCommitHash << std::endl;
    std::cout << "# Built: " << version::kBuildTime << std::endl;
    std::cout << "# Config: " << config_path << std::endl;
    std::cout << "# Input: " << bond_path << std::endl;
    std::cout << "# Spins: " << model.size() << std::endl;
}

void PtColumnNames() {
    std::cout << std::right << std::setw(kHeaderWidth)
        << "Beta" << std::setw(kHeaderWidth)
        << "Samples" << std::setw(kHeaderWidth)
        << "<E>" << std::setw(kHeaderWidth) 
        << "E_MIN" << std::setw(kHeaderWidth) 
        << "P_XCHG" << std::setw(kHeaderWidth) 
        << "Total_Walltime" << std::setw(kHeaderWidth)
        << "Total_Sweeps";
}

void PtResults(ParallelTempering::Result& r) {
    std::cout << std::setprecision(10) << std::scientific << std::setw(kWidth)
        << r.beta << " " << std::setw(kWidth) 
        << r.samples << " " << std::setw(kWidth)
        << r.energy << " " << std::setw(kWidth)
        << r.ground_energy << " " << std::setw(kWidth)
        << r.exchange_probabilty << " " << std::setw(kWidth)
        << r.total_time << " " << std::setw(kWidth)
        << r.total_sweeps << " ";
}

void PtHistograms(std::vector<ParallelTempering::Result>& results) {
    std::cout << std::endl << kMagicString << std::endl << "# No Histograms" << std::endl;
}

// Removed
void IjjDump(Graph& model, std::ostream& stream) {
    stream << std::endl << kMagicString << std::endl << "# Input" << std::endl;
}
}}
