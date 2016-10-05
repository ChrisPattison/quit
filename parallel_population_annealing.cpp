#include "parallel_population_annealing.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <limits>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <functional>
#include <chrono>

void ParallelPopulationAnnealing::CombineHistogram(std::vector<Result::Histogram>& target, const std::vector<Result::Histogram>& source) {
    for(auto bin : source) {
        auto it = std::lower_bound(target.begin(), target.end(), bin, [&](const Result::Histogram& a, const Result::Histogram& b) {return kEpsilon < b.bin - a.bin;});
        if(it==target.end() || bin.bin + kEpsilon <= it->bin) {
            target.insert(it, bin);
        } else {
            it->value += bin.value;
        }
    }
}

ParallelPopulationAnnealing::ParallelPopulationAnnealing(Graph& structure, std::vector<Temperature> betalist, int average_population) : 
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
 
std::vector<ParallelPopulationAnnealing::Result> ParallelPopulationAnnealing::Run() {
    std::vector<Result> results;
    // parallel_.ExecRoot([&](){std::cout << "beta\t<E>\t \tR\t \tE_MIN\t \tR_MIN\tR_MIN/R\t \tS\tR/e^S" << std::endl;});
    
    for(auto& r : replicas_) {
        for(std::size_t k = 0; k < r.size(); ++k) {
            r(k) = rng_.Get<bool>() ? 1 : -1;
        }
    }

    std::iota(replica_families_.begin(), replica_families_.end(), average_node_population_ * parallel_.rank());
    beta_ = betalist_.at(0).beta;
    const int M = 10;
    const int max_family_size_limit = average_node_population_ / 2;
    Eigen::VectorXd energy;

    for(auto new_beta : betalist_) {
        Result observables;
        
        auto time_start = std::chrono::high_resolution_clock::now();
        if(new_beta.beta != beta_) {
            Resample(new_beta.beta);
            auto redist_time_start = std::chrono::high_resolution_clock::now();
            Redistribute();
            observables.redist_walltime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - redist_time_start).count();
            for(std::size_t k = 0; k < replicas_.size(); ++k) {
                MonteCarloSweep(replicas_[k], M);
            }
        }else {
            observables.redist_walltime = 0.0;
        }
        observables.montecarlo_walltime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - time_start).count();
        time_start = std::chrono::high_resolution_clock::now();

        energy.resize(replicas_.size());
        for(std::size_t k = 0; k < replicas_.size(); ++k) {
            energy(k) = Hamiltonian(replicas_[k]);
        }
        
        // Observables
        observables.beta = beta_;

        // population
        observables.population = parallel_.HeirarchyReduceToAll<int>(static_cast<int>(replicas_.size()), 
            [](std::vector<int>& v) { return std::accumulate(v.begin(), v.end(), 0.0, std::plus<int>()); });

        // average energy
        observables.average_energy = parallel_.HeirarchyReduce<double>(energy.array().mean(),
            [](std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0, std::plus<double>()); }) / parallel_.size();

        // ground energy
        observables.ground_energy = parallel_.HeirarchyReduceToAll<double>(energy.minCoeff(), 
            [](std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); });

        // Round-off /probably/ isn't an issue here. Make this better in the future
        // number of replicas with energy = ground energy
        observables.grounded_replicas = parallel_.HeirarchyReduce<double>(
            energy.array().unaryExpr([&](double E) { return E == observables.ground_energy ? 1 : 0; }).sum(),
            [](std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0, std::plus<double>()); });

        // Largest Family
        std::vector<double> family_size = FamilyCount();
        observables.max_family_size = *std::max_element(family_size.begin(), family_size.end());
        // Entropy
        std::transform(family_size.begin(), family_size.end(), family_size.begin(), 
            [&](double n) -> double { n /= observables.population; return n * std::log(n); });
        observables.entropy = parallel_.HeirarchyReduce<double>(-std::accumulate(family_size.begin(), family_size.end(), 0.0), 
            [](std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0, std::plus<double>()); });
        
        if(new_beta.histograms) {            
            // Overlap
            std::vector<std::pair<int, int>> overlap_pairs = BuildReplicaPairs();
            std::vector<double> overlap_samples(overlap_pairs.size());

            std::transform(overlap_pairs.begin(), overlap_pairs.end(), overlap_samples.begin(),
                [&](std::pair<int, int> p) { return Overlap(replicas_[p.first], replicas_[p.second]); });
            observables.overlap = parallel_.HeirarchyVectorReduce<Result::Histogram>(BuildHistogram(overlap_samples), 
                [&](std::vector<Result::Histogram>& accumulator, const std::vector<Result::Histogram>& value) { CombineHistogram(accumulator, value); });
            std::transform(observables.overlap.begin(), observables.overlap.end(), observables.overlap.begin(),
                [&](Result::Histogram v) -> Result::Histogram {
                    v.value /= parallel_.size(); 
                    return v;
                });
            // Link Overlap
            std::transform(overlap_pairs.begin(), overlap_pairs.end(), overlap_samples.begin(),
                [&](std::pair<int, int> p) { return LinkOverlap(replicas_[p.first], replicas_[p.second]); });
            observables.link_overlap = parallel_.HeirarchyVectorReduce<Result::Histogram>(BuildHistogram(overlap_samples), 
                [&](std::vector<Result::Histogram>& accumulator, const std::vector<Result::Histogram>& value) { CombineHistogram(accumulator, value); });
            std::transform(observables.link_overlap.begin(), observables.link_overlap.end(), observables.link_overlap.begin(), 
                [&](Result::Histogram v) -> Result::Histogram {
                    v.value /= parallel_.size(); 
                    return v;
                });
        }

        char family_size_exceeded = parallel_.HeirarchyReduceToAll<char>(observables.max_family_size > max_family_size_limit, 
            [](std::vector<char>& v) { return std::accumulate(v.begin(), v.end(), 0, [](const char& lhs, const char& rhs) { return (lhs | rhs) != 0 ? 1 : 0; }); });

        observables.observables_walltime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - time_start).count();
        
        parallel_.ExecRoot([&](){
            results.push_back(observables);
            // std::cout 
            //     << observables.beta << ",\t" 
            //     << observables.average_energy << ",\t" 
            //     << observables.population << ",\t \t" 
            //     << observables.ground_energy << ",\t" 
            //     << observables.grounded_replicas << ",\t" 
            //     << static_cast<double>(observables.grounded_replicas)/observables.population << ",\t \t" 
            //     << observables.entropy << ",\t" 
            //     << observables.population/std::exp(observables.entropy) << std::endl; 
        });

        if(family_size_exceeded) {
            break;
        }
    }
    return results;
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

    double summed_weights = parallel_.HeirarchyReduceToAll<double>(weighting.sum(), [] (std::vector<double>& v) {
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

void ParallelPopulationAnnealing::Redistribute() {
    struct rank_population {
        int rank;
        int population;
    };

    std::vector<int> populations = parallel_.AllGather(static_cast<int>(replicas_.size()));
    std::vector<rank_population> node_populations(populations.size());
    for(auto p = 0; p < node_populations.size(); ++p) {
        node_populations[p] = {p, populations[p]};
    }
    // Sort population list in ascending order by population and find number of nodes that exceed the population limit
    std::sort(node_populations.begin(), node_populations.end(), 
        [](const rank_population& a, const rank_population& b) {return a.population < b.population; });
    int redist_nodes = std::distance(std::lower_bound(node_populations.begin(), node_populations.end(), static_cast<int>(average_node_population_ * kMaxPopulation),
        [](const rank_population& a, const int& b) {return a.population < b; }), node_populations.end());
    redist_nodes = std::min(redist_nodes, parallel_.size() / 2);
    // Position of current rank and the complementary rank in population list
    auto position = std::find_if(node_populations.begin(), node_populations.end(), [&](const rank_population& a) { return a.rank == parallel_.rank(); });
    auto complement_position = --(node_populations.end()) + -std::distance(node_populations.begin(), position);
    // receiving replicas
    if(std::distance(node_populations.begin(), position) < redist_nodes) { 
        int source = complement_position->rank;
        std::vector<int> packed_replicas = parallel_.Receive<std::vector<int>>(source);
        std::vector<int> packed_families = parallel_.Receive<std::vector<int>>(source);
        int pack_size = packed_families.size();
        // Unpack
        replicas_.reserve(replicas_.size() + pack_size);
        for(int k = 0; k < pack_size; ++k) {
            replicas_.push_back(Eigen::Map<StateVector>(&(packed_replicas[k*structure_.size()]), structure_.size()));
        }
        replica_families_.insert(replica_families_.end(), packed_families.begin(), packed_families.end());
    // sending replicas
    }else if(std::distance(node_populations.begin(), complement_position) < redist_nodes) { 
        std::vector<int> packed_replicas;
        std::vector<int> packed_families;

        int pack_size = replicas_.size() - average_node_population_;
        auto pack_start_candidate = replica_families_.begin() + rng_.Range(replicas_.size() - pack_size);
        int pack_start = std::distance(replica_families_.begin(), std::find_if(pack_start_candidate, replica_families_.end(), 
            [&](const int& a) { return a != *pack_start_candidate; }));
        // align to family boundary
        // TODO: look into if breaking up families is preferrable
        pack_size = std::distance(replica_families_.begin() + pack_start, std::find_if(replica_families_.begin() + std::min(static_cast<std::size_t>(pack_start + pack_size), replica_families_.size()), replica_families_.end(),
            [&](const int& a) { return a != replica_families_[pack_start + pack_size]; }));
        packed_replicas.reserve(pack_size * structure_.size());
        // Pack
        for(int k = 0; k < pack_size; ++k) {
            auto& r = replicas_[pack_start + k];
            packed_replicas.insert(packed_replicas.end(), r.data(), r.data() + r.size());
        }
        packed_families.insert(packed_families.end(), replica_families_.begin() + pack_start, replica_families_.begin() + pack_start + pack_size);
        // Erase sent replicas
        replica_families_.erase(replica_families_.begin() + pack_start, replica_families_.begin() + pack_start + pack_size);
        replicas_.erase(replicas_.begin() + pack_start, replicas_.begin() + pack_start + pack_size);

        int target = (complement_position)->rank;
        parallel_.Send<std::vector<int>>(packed_replicas, target);
        parallel_.Send<std::vector<int>>(packed_families, target);
    }
}