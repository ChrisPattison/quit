#include "fpga_population_annealing.hpp"
#include "graph.hpp"
#include <vector>

void FpgaPopulationAnnealing::MonteCarloSweep(StateVector& replica, int moves) {
    // Pack bit vector for now...
    std::vector<std::uint32_t> packed_replica((replica.size() + 31)/32);
    for(int i = 0; i < replica.size(); ++i) {
        int shift = i % 32;
        packed_replica[i/32] = (packed_replica[i/32] & ((~0-1) << shift)) | (static_cast<std::uint32_t>(replica[i]==-1 ? 0 : 1) << shift);
    }

    driver_.Sweep(&packed_replica, static_cast<std::uint32_t>(moves));

    for(int i = 0; i < replica.size(); ++i) {
        int shift = i % 32;
        replica[i] = packed_replica[i/32] & (1 << shift) ? 1 : -1;
    }
}

double FpgaPopulationAnnealing::Resample(double new_beta) {
    PopulationAnnealing::Resample(new_beta);
    driver_.SetProb(beta_);
}

FpgaPopulationAnnealing::FpgaPopulationAnnealing(Graph& structure, Config schedule) : PopulationAnnealing(structure, schedule) {
    driver_.SeedRng(rng_.GetSeed());
    driver_.SetGraph(structure_);
    driver_.SetProb(beta_);
}