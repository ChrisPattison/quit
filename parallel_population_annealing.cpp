#include "parallel_population_annealing.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <limits>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <functional>

ParallelPopulationAnnealing::ParallelPopulationAnnealing(Graph& structure, std::vector<double> betalist, int average_population) : 
        PopulationAnnealing(structure, betalist, 0) {
        
    average_node_population_ = average_population / parallel_.size();
    average_population_ = average_node_population_ * parallel_.size();
    rng_ = RandomNumberGenerator(parallel_.rank());

    replicas_.resize(average_node_population_);
    replica_families_.resize(average_node_population_);
    for(auto& r : replicas_) {
        r = StateVector();
        r.resize(structure_.size());
    }
 }

void ParallelPopulationAnnealing::Run(std::vector<Result>& results) {
    parallel_.ExecRoot([&](){std::cout << "beta\t<E>\t \tR\t \tE_MIN\t \tR_MIN\tR_MIN/R\t \tS\tR/e^S" << std::endl;});
    
    for(auto& r : replicas_) {
        for(std::size_t k = 0; k < r.size(); ++k) {
            r(k) = rng_.Get<bool>() ? 1 : -1;
        }
    }

    std::iota(replica_families_.begin(), replica_families_.end(), replicas_.size() * parallel_.rank());
    beta_ = betalist_.at(0);
    int M = 10;
    Eigen::VectorXd energy;

    for(auto new_beta : betalist_) {
        Result observables;
        double H;
        Resample(new_beta);
        energy.resize(replicas_.size());

        for(std::size_t k = 0; k < replicas_.size(); ++k) {
            MonteCarloSweep(replicas_[k], M);
            H = Hamiltonian(replicas_[k]);
            energy(k) = H;
        }
        
        // Observables
        observables.beta = beta_;

        // population
        observables.population = parallel_.Reduce<std::size_t>(replicas_.size(), 
            [](std::vector<std::size_t>& v) { return std::accumulate(v.begin(), v.end(), 0.0, std::plus<std::size_t>()); });

        // average energy
        observables.average_energy = parallel_.Reduce<double>(energy.array().mean(),
            [](std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0, std::plus<double>()); }) / parallel_.size();

        // ground energy
        observables.ground_energy = parallel_.ReduceToAll<double>(energy.minCoeff(), 
            [](std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); });

        // Round-off /probably/ isn't an issue here. Make this better in the future
        // number of replicas with energy = ground energy
        observables.grounded_replicas = parallel_.Reduce<double>(
            energy.array().unaryExpr([&](double E) { return E == observables.ground_energy ? 1 : 0; }).sum(),
            [](std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0, std::plus<double>()); });

        // Entropy
        std::vector<double> family_size = FamilyCount();
        std::transform(family_size.begin(), family_size.end(), family_size.begin(), 
            [&](double n) { return n * std::log(n); });
        observables.entropy = -parallel_.Reduce<double>(std::accumulate(family_size.begin(), family_size.end(), 0.0), 
            [](std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0, std::plus<double>()); });
        
        // Overlap -- do this later
        // OverlapPmd(observables.overlap);
        
        parallel_.ExecRoot([&](){
            results.push_back(observables);
            std::cout 
                << observables.beta << ",\t" 
                << observables.average_energy << ",\t" 
                << observables.population << ",\t \t" 
                << observables.ground_energy << ",\t" 
                << observables.grounded_replicas << ",\t" 
                << static_cast<double>(observables.grounded_replicas)/observables.population << ",\t \t" 
                << observables.entropy << ",\t" 
                << observables.population/std::exp(observables.entropy) << std::endl; 
        });
    }
}

void ParallelPopulationAnnealing::Resample(double new_beta) {
    std::vector<StateVector> resampled_replicas;
    std::vector<int> resampled_families;
    resampled_replicas.reserve(replicas_.size());
    resampled_families.reserve(replicas_.size());
    
    Eigen::VectorXd weighting(replicas_.size());
    for(std::size_t k = 0; k < replicas_.size(); ++k) {
        weighting(k) = std::exp(-(new_beta-beta_) * Hamiltonian(replicas_[k]));
    }

    double summed_weights = parallel_.ReduceToAll<double>(weighting.sum(), [] (std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0, std::plus<double>());
    });
    
    double normalize = average_population_ / summed_weights;
    for(std::size_t k = 0; k < replicas_.size(); ++k) {
        double weight = normalize * weighting(k);
        unsigned int n = (weight - std::floor(weight)) > rng_.Probability() ? std::ceil(weight) : std::floor(weight);
        for(std::size_t i = 0; i < n; ++i) {
            resampled_replicas.push_back(replicas_[k]);
            resampled_families.push_back(replica_families_[k]);
        }
    }

    beta_ = new_beta;
    replicas_ = resampled_replicas;
    replica_families_ = resampled_families;
}
