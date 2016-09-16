#pragma once
#include <vector>
#include <utility>
#include "types.hpp"
#include "graph.hpp"
#include "types.hpp"
#include "random_number_generator.hpp"
// #include "population_annealing.hpp"
#include "parallel.hpp"
#include "Eigen/Dense"

class ParallelPopulationAnnealing {
public:
    struct Result {
        struct ProbabilityMass {
            double bin;
            double mass;
        };
        std::vector<ProbabilityMass> overlap;
        double beta;
        std::size_t population;
        double average_energy;
        double ground_energy;
        std::size_t grounded_replicas;
        double entropy;
    };

    
private:

    Parallel parallel_;

    Graph structure_;

    RandomNumberGenerator rng_;

    using StateVector = Eigen::Matrix<VertexType, Eigen::Dynamic, 1>;
    std::vector<StateVector> replicas_;
    std::vector<int> replica_families_;

    int average_node_population_;
    std::vector<double> betalist_;
    double beta_;

    // Builds a list of replicas with different Markov Chains
    std::vector<std::pair<int, int>> BuildReplicaPairs();
    // Gets the number of replicas in each family as a fraction of the total population
    std::vector<double> FamilyCount();

    double AcceptanceProbability(double delta_energy) const;
    // Gives the Hamiltonian of the given state
    double Hamiltonian(StateVector& replica);
    // Change in energy associated with a single change in state
    double DeltaEnergy(StateVector& replica, int vertex); 

    void MonteCarloSweep(StateVector& replica, int moves);

    bool IsLocalMinimum(StateVector& replica);

    StateVector Quench(const StateVector& replica);

    double Overlap(StateVector& alpha, StateVector& beta);

    void OverlapPmd(std::vector<Result::ProbabilityMass>& pmd);

    void Resample(double beta);
public:

    ParallelPopulationAnnealing() = delete;

    ParallelPopulationAnnealing(const ParallelPopulationAnnealing&) = delete;

    explicit ParallelPopulationAnnealing(Graph& structure, std::vector<double> betalist, int average_population);

    void Run(std::vector<Result>& results);
};