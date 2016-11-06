#pragma once
#include <vector>
#include <utility>
#include "types.hpp"
#include "graph.hpp"
#include "types.hpp"
#include "random_number_generator.hpp"
#include "population_annealing.hpp"
#include "parallel.hpp"
#include <Eigen/Dense>

class ParallelPopulationAnnealing : protected PopulationAnnealing {
protected:
    static constexpr double kMaxPopulation = 1.10;

    parallel::Mpi parallel_;

    int average_node_population_;

    double Resample(double new_beta);

    void Redistribute();

    void CombineHistogram(std::vector<Result::Histogram>& target, const std::vector<Result::Histogram>& source);

    std::vector<double> FamilyCount();

    std::vector<VertexType> Pack(const std::vector<StateVector>& source);

    std::vector<VertexType> Pack(const std::vector<StateVector>::const_iterator begin_iterator, const std::vector<StateVector>::const_iterator end_iterator);

    std::vector<StateVector> Unpack(std::vector<VertexType>& source);
public:

    struct Result : PopulationAnnealing::Result {
        double montecarlo_walltime;
        double redist_walltime;
        double observables_walltime;
        int max_family_size;
    };

    ParallelPopulationAnnealing(const ParallelPopulationAnnealing&) = delete;

    ParallelPopulationAnnealing(Graph& structure, std::vector<Schedule> schedule, int average_population);

    std::vector<Result> Run();
};