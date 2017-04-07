#include "fpga_population_annealing.hpp"
#include "graph.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

void FpgaPopulationAnnealing::Sweep(StateVector& replica, int moves) {
    // Pack bit vector for now...
    std::vector<std::uint32_t> packed_replica((replica.size() + 31)/32);

    for(int j = 0; j < packed_replica.size(); ++j) {
        packed_replica[j] = 0;
        for(int i = 0; i < 32; ++i) {
            int spin = j*32 + i;
            packed_replica[j] >>= 1;
            if(spin < replica.size() && replica(spin) == 1) {
                packed_replica[j] |= 0x80000000;
            }
        }
    }

    driver_.Sweep(&packed_replica, static_cast<std::uint32_t>(moves));

    for(int i = 0; i < replica.size(); ++i) {
        int shift = i % 32;
        replica[i] = packed_replica[i/32] & (1 << shift) ? 1 : -1;
    }
}

double FpgaPopulationAnnealing::Resample(double new_beta) {
    driver_.SetProb(new_beta);
    return PopulationAnnealing::Resample(new_beta);
}

FpgaPopulationAnnealing::FpgaPopulationAnnealing(Graph& structure, Config schedule) : PopulationAnnealing(structure, schedule) {
    driver_.SeedRng(rng_.GetSeed());
    driver_.SetGraph(structure_);
    driver_.SetProb(0.0);
}
