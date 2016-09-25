#pragma once
#include <vector>
#include <utility>
#include "types.hpp"
#include "graph.hpp"
#include "types.hpp"
#include "random_number_generator.hpp"
#include "population_annealing.hpp"
#include "parallel.hpp"
#include "Eigen/Dense"

class ParallelPopulationAnnealing : protected PopulationAnnealing {
protected:

    Parallel parallel_;

    int average_node_population_;

    void Resample(double new_beta);

    std::vector<Result::Histogram> CombineHistogram(const std::vector<std::vector<Result::Histogram>>& histograms);

public:

    using Result = PopulationAnnealing::Result;

    ParallelPopulationAnnealing(const ParallelPopulationAnnealing&) = delete;

    ParallelPopulationAnnealing(Graph& structure, std::vector<double> betalist, int average_population);

    void Run(std::vector<Result>& results);
};