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
    
    if(parallel_.size() / 2 * 2 != parallel_.size()) {
        return results;
    }

    for(auto& r : replicas_) {
        for(std::size_t k = 0; k < r.size(); ++k) {
            r(k) = rng_.Get<bool>() ? 1 : -1;
        }
    }

    std::iota(replica_families_.begin(), replica_families_.end(), average_node_population_ * parallel_.rank());
    beta_ = betalist_.at(0).beta;
    const int M = 10;
    const int max_family_size_limit = average_population_ / 2;
    Eigen::VectorXd energy;

    for(auto new_beta : betalist_) {
        Result observables;
        auto time_start = std::chrono::high_resolution_clock::now();
        // Population Annealing
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
        // This breaks if all the population belonging to a node is killed off at once
        // Observables
        observables.beta = beta_;

        // population
        observables.population = parallel_.HeirarchyReduceToAll<int>(static_cast<int>(replicas_.size()), 
            [](std::vector<int>& v) { return std::accumulate(v.begin(), v.end(), 0.0, std::plus<int>()); });

        // average energy
        observables.average_energy = parallel_.HeirarchyReduce<double>(energy.size() ? energy.array().mean() : std::numeric_limits<double>::quiet_NaN(),
            [](std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0, std::plus<double>()); }) / parallel_.size();

        // ground energy
        observables.ground_energy = parallel_.HeirarchyReduceToAll<double>(energy.size() ? energy.minCoeff() : std::numeric_limits<double>::quiet_NaN(), 
            [](std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); });

        // smallest node population
        observables.min_node_population = 0;/*parallel_.HeirarchyReduceToAll<int>(static_cast<int>(replicas_.size()),
            [](std::vector<int>& v) { return *std::min_element(v.begin(), v.end()); });*/

        // Round-off /probably/ isn't an issue here. Make this better in the future
        // number of replicas with energy = ground energy
        observables.grounded_replicas = parallel_.HeirarchyReduce<double>(
            energy.size() ? energy.array().unaryExpr([&](double E) { return E == observables.ground_energy ? 1 : 0; }).sum() : 0,
            [](std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0, std::plus<double>()); });

        // Largest Family
        std::vector<double> family_size = FamilyCount();
        observables.max_family_size = family_size.size() > 0 ? *std::max_element(family_size.begin(), family_size.end()) : 0;
        // Entropy
        std::transform(family_size.begin(), family_size.end(), family_size.begin(), 
            [&](double n) -> double { n /= observables.population; return n * std::log(n); });
        observables.entropy = parallel_.HeirarchyReduce<double>(-std::accumulate(family_size.begin(), family_size.end(), 0.0), 
            [](std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0, std::plus<double>()); });
        
        if(new_beta.histograms) {
            // Import or Export replicas to complementary node
            std::vector<StateVector> imported_replicas;
            if(parallel_.rank() < parallel_.size()/2) {
                std::vector<VertexType> replica_pack = parallel_.Receive<std::vector<VertexType>>(parallel_.rank() + parallel_.size()/2);
                imported_replicas = Unpack(replica_pack);
            }else {
                std::vector<VertexType> replica_pack = Pack(replicas_);
                parallel_.Send(replica_pack, parallel_.rank() - parallel_.size()/2);
            }

            int sample_size = std::min(replicas_.size(), imported_replicas.size()); 
            std::vector<double> overlap_samples(sample_size);
            
            // Overlap
            for(int k = 0; k < sample_size; ++k) {
                overlap_samples[k] = Overlap(replicas_[k], imported_replicas[k]);
            }
            observables.overlap = parallel_.HeirarchyVectorReduce<Result::Histogram>(BuildHistogram(overlap_samples), 
                [&](std::vector<Result::Histogram>& accumulator, const std::vector<Result::Histogram>& value) { CombineHistogram(accumulator, value); });
            std::transform(observables.overlap.begin(), observables.overlap.end(), observables.overlap.begin(),
                [&](Result::Histogram v) -> Result::Histogram {
                    v.value /= parallel_.size(); 
                    return v;
                });

            // Link Overlap
            for(int k = 0; k < sample_size; ++k) {
                overlap_samples[k] = LinkOverlap(replicas_[k], imported_replicas[k]);
            }
            observables.link_overlap = parallel_.HeirarchyVectorReduce<Result::Histogram>(BuildHistogram(overlap_samples), 
                [&](std::vector<Result::Histogram>& accumulator, const std::vector<Result::Histogram>& value) { CombineHistogram(accumulator, value); });
            std::transform(observables.link_overlap.begin(), observables.link_overlap.end(), observables.link_overlap.begin(), 
                [&](Result::Histogram v) -> Result::Histogram {
                    v.value /= parallel_.size(); 
                    return v;
                });
        }

        observables.max_family_size = parallel_.HeirarchyReduceToAll<int>(observables.max_family_size, 
            [](std::vector<int>& v) { return *std::max_element(v.begin(), v.end()); });

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

        if(observables.max_family_size > max_family_size_limit) {
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

// TODO: implement reverse transfer, maybe higher order terms in transfer rate
void ParallelPopulationAnnealing::Redistribute() {
    struct rank_population {
        int rank;
        int population;
    };

    struct rank_package {
        int rank;
        std::vector<VertexType> replicas;
        std::vector<int> families;
        parallel::AsyncOp<VertexType> send_replicas_op;
        parallel::AsyncOp<int> send_families_op;
    };

    std::vector<int> populations = parallel_.AllGather(static_cast<int>(replicas_.size()));
    std::vector<rank_population> node_populations(populations.size());
    for(auto p = 0; p < node_populations.size(); ++p) {
        node_populations[p] = {p, populations[p]};
    }
    // Position of current rank and the targets in population list
    auto position = node_populations.begin() + parallel_.rank();
    auto send_target = position != node_populations.end() - 1 ? position + 1 : node_populations.begin();
    auto recv_source = position != node_populations.begin() ? position - 1 : node_populations.end() - 1;
    // sending replicas
    std::vector<rank_package> packages;

    if(position->population > average_node_population_ * kMaxPopulation ||
    send_target->population < average_node_population_ * kMinPopulation) {
        // Get all nodes with 0 population infront of current node
        do {
            packages.emplace_back();
            packages.back().rank = send_target->rank;
            send_target++;
        }while(send_target->population == 0);

        // Compute pack size (limited gradient)
        // This may need some "Dampening" to keep a large population wave from moving through the nodes
        int pack_size = std::min(std::min(
            std::max(0, (position->population - send_target->population)/2),
            std::max(0, position->population - static_cast<int>(average_node_population_ * kMinPopulation))),
            static_cast<int>(replicas_.size()/2)) / packages.size();
        for(auto pack = packages.rbegin(); pack != packages.rend(); pack++) {
            int pack_start = replicas_.size() - pack_size;
            pack->replicas.reserve(pack_size * structure_.size());

            // Pack
            for(int k = 0; k < pack_size; ++k) {
                auto& r = replicas_[pack_start + k];
                pack->replicas.insert(pack->replicas.end(), r.data(), r.data() + r.size());
            }
            pack->families.insert(pack->families.end(), replica_families_.begin() + pack_start, replica_families_.begin() + pack_start + pack_size);
            
            // Erase sent replicas
            replica_families_.erase(replica_families_.begin() + pack_start, replica_families_.begin() + pack_start + pack_size);
            replicas_.erase(replicas_.begin() + pack_start, replicas_.begin() + pack_start + pack_size);

            pack->send_replicas_op = parallel_.SendAsync(&pack->replicas, pack->rank);
            pack->send_families_op = parallel_.SendAsync(&pack->families, pack->rank);
        }
    }
    // receiving replicas
    if(recv_source->population > average_node_population_ * kMaxPopulation ||
    position->population < average_node_population_ * kMinPopulation) {
        if(position->population == 0) {
            while(recv_source->population == 0) {
                recv_source++;
            }
        }
        std::vector<VertexType> packed_replicas = parallel_.Receive<std::vector<VertexType>>(recv_source->rank);
        std::vector<int> packed_families = parallel_.Receive<std::vector<int>>(recv_source->rank);

        assert(packed_families.size() == packed_replicas.size() / structure_.size());
        int pack_size = packed_families.size();
        // Unpack
        replicas_.reserve(replicas_.size() + pack_size);
        replicas_.insert(replicas_.begin(), pack_size, {});
        for(int k = 0; k < pack_size; ++k) {
            replicas_[k] = Eigen::Map<StateVector>(&(packed_replicas[k*structure_.size()]), structure_.size());
        }
        replica_families_.insert(replica_families_.begin(), packed_families.begin(), packed_families.end());
    }
}

std::vector<VertexType> ParallelPopulationAnnealing::Pack(const std::vector<StateVector>& source) {
    std::vector<VertexType> destination;
    destination.reserve(source.size() * structure_.size());

    for(int k = 0; k < source.size(); ++k) {
        auto& r = source[k];
        destination.insert(destination.end(), r.data(), r.data() + r.size());
    }
    return destination;
}

std::vector<PopulationAnnealing::StateVector> ParallelPopulationAnnealing::Unpack(std::vector<VertexType>& source) {
    assert(source.size() % structure_.size() == 0);
    int pack_size = source.size() / structure_.size();

    std::vector<StateVector> destination;
    destination.reserve(pack_size);
    for(int k = 0; k < pack_size; ++k) {
        destination.push_back(Eigen::Map<StateVector>(&(source[k*structure_.size()]), structure_.size()));
    }
    return destination;
}

// This suffers from problems if there are non-contiguous families on the owner's node
std::vector<double> ParallelPopulationAnnealing::FamilyCount() {
    struct Family {
        int tag;
        int count;
    };
    std::vector<Family> families;
    std::vector<Family> local_families;
    families.reserve(replicas_.size());

    // count families
    auto i = replica_families_.begin();
    do {
        auto i_next = std::find_if(i+1, replica_families_.end(), [&](const int& v){return v != *i;});
        families.push_back({*i, std::distance(i, i_next)});
        i = i_next;
    }while(i != replica_families_.end());
    // package for sending to originator
    local_families.reserve(families.size());
    std::vector<parallel::Packet<Family>> packets;
    
    for(auto& f : families) {
        int source_rank = f.tag / average_node_population_;
        if(source_rank != parallel_.rank()) {
            auto it = std::find_if(packets.begin(), packets.end(), [&](const parallel::Packet<Family> p) { return p.rank == source_rank; });
            if(it != packets.end()) {
                it->data.push_back(f);
            }else {
                packets.push_back({source_rank, { f }});
            }
        }else {
            local_families.push_back(f);
        }
    }
    // send and count recieved families
    auto it = local_families.begin();
    std::vector<Family> import_families = parallel_.PartialReduce(packets);
    for(auto& f : import_families) {
        // find iterator to matching family with optimization for successive families
        if(!(++it < local_families.end() && it->tag == f.tag)) {
            it = std::find_if(local_families.begin(), local_families.end(), [&](const Family& match) { return match.tag == f.tag; });
        }
        
        if(it == local_families.end()) {
            local_families.push_back(f);
        }else {
            it->count += f.count;
        }
    }

    std::vector<double> result(local_families.size());
    std::transform(local_families.begin(), local_families.end(), result.begin(), [](const Family& f) { return static_cast<double>(f.count); });
    return result;
}