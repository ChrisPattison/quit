#include "fpga_population_annealing.hpp"
#include "graph.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

void FpgaPopulationAnnealing::Sweep(int moves) {
}

double FpgaPopulationAnnealing::Resample(double new_beta, double population_fraction) {
    driver_.SetProb(new_beta);
    return PopulationAnnealing::Resample(new_beta, population_fraction);
}

FpgaPopulationAnnealing::FpgaPopulationAnnealing(Graph& structure, Config schedule) : PopulationAnnealing(structure, schedule) {
    driver_.SeedRng(rng_.GetSeed());
    driver_.SetGraph(structure_);
    driver_.SetProb(0.0);
}
