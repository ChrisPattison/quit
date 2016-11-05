#include "population_annealing.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <limits>
#include <algorithm>
#include <iterator>
#include <numeric>

// Fix this to do something with the single replicas
std::vector<std::pair<int, int>> PopulationAnnealing::BuildReplicaPairs() {
    std::vector<std::pair<int, int>> pairs;
    int num_pairs = replica_families_.size()/2;
    pairs.reserve(num_pairs);

    for(int k = 0; k < num_pairs; ++k) {
        pairs.push_back({k, k + num_pairs});
    }

    // Consistency check

    for(auto& it : pairs) {
        // this is highly unlikely to be true
        if(replica_families_[it.first] == replica_families_[it.second]) {
            for(auto& it_other : pairs) {
                if(replica_families_[it_other.first] != replica_families_[it.second] && 
                    replica_families_[it.first] != replica_families_[it_other.second]) {
                    it.swap(it_other);
                    break;
                }
            }
        }
    }
    return pairs;
}

std::vector<PopulationAnnealing::Result::Histogram> PopulationAnnealing::BuildHistogram(const std::vector<double>& samples) {
    std::vector<Result::Histogram> hist;
    for(auto v : samples) {
        auto it = std::lower_bound(hist.begin(), hist.end(), v, [&](const Result::Histogram& a, const double& b) {return kEpsilon < b - a.bin;});
        if(it==hist.end() || v + kEpsilon <= it->bin) {
            hist.insert(it, {v, 1.0});
        } else {
            it->value++;
        }
    }
    for(auto& bin : hist) {
        bin.value /= samples.size();
    }
    return hist;
}

std::vector<double> PopulationAnnealing::FamilyCount() {
    std::vector<double> count;
    count.reserve(replica_families_.size());
    auto i = replica_families_.begin();
    do {
        auto i_next = std::find_if(i, replica_families_.end(), [&](const int& v){return v != *i;});
        count.push_back(static_cast<double>(std::distance(i, i_next)));
        i = i_next;
    }while(i != replica_families_.end());
    return count;
}

PopulationAnnealing::PopulationAnnealing(Graph& structure, std::vector<Temperature> betalist, int average_population) {
    betalist_ = betalist;
    beta_ = NAN;
    structure_ = structure;
    structure_.Adjacent().makeCompressed();
    average_population_ = average_population;
    replicas_.resize(average_population_);
    replica_families_.resize(average_population_);
    for(auto& r : replicas_) {
        r = StateVector();
        r.resize(structure_.size());
    }
    log_lookup_table_.resize(lookup_table_size_);
    for(int k = 0; k < log_lookup_table_.size(); ++k) {
        log_lookup_table_[k] = std::log(static_cast<double>(k) / (log_lookup_table_.size() - 1));
    }
 }

double PopulationAnnealing::Hamiltonian(StateVector& replica) {
    return ((structure_.Adjacent().triangularView<Eigen::Upper>() * replica.cast<EdgeType>()).array() * replica.cast<EdgeType>().array()).sum();
}

double PopulationAnnealing::DeltaEnergy(StateVector& replica, int vertex) {
    double h = structure_.Adjacent().innerVector(vertex).dot(replica.cast<EdgeType>());
    // h -= structure_.Fields()(vertex);
    return -2 * replica(vertex) * h;
}

void PopulationAnnealing::MonteCarloSweep(StateVector& replica, int sweeps) {
    for(std::size_t k = 0; k < sweeps; ++k) {
        for(std::size_t i = 0; i < replica.size(); ++i) {
            int vertex = rng_.Range(replica.size());
            double delta_energy = DeltaEnergy(replica, vertex);
            
            //round-off isn't a concern here
            if(AcceptedMove(delta_energy)) {
                replica(vertex) *= -1;
            }
        }
    }
}

bool PopulationAnnealing::IsLocalMinimum(StateVector& replica) {
    for(int k = 0; k < replica.size(); ++k) {
        if(DeltaEnergy(replica, k) < 0) {
            return false;
        }
    }
    return true;
}

PopulationAnnealing::StateVector PopulationAnnealing::Quench(const StateVector& replica) {
    const std::size_t sweeps = 4;
    StateVector quenched_replica = replica;
    do {
        for(std::size_t k = 0; k < sweeps * quenched_replica.size(); ++k) {
            int vertex = rng_.Range(quenched_replica.size());
            if(DeltaEnergy(quenched_replica, vertex) < 0) {
                quenched_replica(vertex) *= -1;
            }
        }
    }while(!IsLocalMinimum(quenched_replica));
    return quenched_replica;
}

double PopulationAnnealing::Overlap(StateVector& alpha, StateVector& beta) {
    return (alpha.array() * beta.array()).cast<double>().sum() / structure_.size();
}

double PopulationAnnealing::LinkOverlap(StateVector& alpha, StateVector& beta) {
    double ql = 0;
    for(std::size_t k = 0; k < structure_.Adjacent().outerSize(); ++k) {
        for(Eigen::SparseTriangularView<Eigen::SparseMatrix<EdgeType>,Eigen::Upper>::InnerIterator 
            it(structure_.Adjacent().triangularView<Eigen::Upper>(), k); it; ++it) {
            ql += alpha(k) * beta(k) * alpha(it.index()) * beta(it.index());
        }
    }
    return ql / structure_.edges();
}

std::vector<PopulationAnnealing::Result> PopulationAnnealing::Run() {
    std::vector<Result> results;
    
    for(auto& r : replicas_) {
        for(std::size_t k = 0; k < r.size(); ++k) {
            r(k) = rng_.Probability() < 0.5 ? 1 : -1;
        }
    }

    std::iota(replica_families_.begin(), replica_families_.end(), 0);
    beta_ = betalist_.at(0).beta;
    int M = 10;
    std::vector<double> energy;

    for(auto new_beta : betalist_) {
        Result observables;
        
        if(new_beta.beta != beta_) {
            observables.norm_factor = Resample(new_beta.beta);
            for(std::size_t k = 0; k < replicas_.size(); ++k) {
                MonteCarloSweep(replicas_[k], M);
            }
        }

        energy.resize(replicas_.size());
        for(std::size_t k = 0; k < replicas_.size(); ++k) {
            energy[k] = Hamiltonian(replicas_[k]);
        }
        Eigen::Map<Eigen::VectorXd> energy_map(energy.data(), energy.size());
        // Basic observables
        observables.beta = beta_;
        observables.population = replicas_.size();
        observables.average_energy = energy_map.mean();
        observables.average_energy_squared = energy_map.array().pow(2).mean();
        observables.ground_energy = energy_map.minCoeff();
        // Round-off /probably/ isn't an issue here
        observables.grounded_replicas = energy_map.array().unaryExpr(
            [&](double E){return E == observables.ground_energy ? 1 : 0;}).sum();
        // Entropy
        std::vector<double> family_size = FamilyCount();
        std::transform(family_size.begin(),family_size.end(),family_size.begin(),
            [&](double n) -> double {n /= replicas_.size(); return n * std::log(n);});
        observables.entropy = -std::accumulate(family_size.begin(), family_size.end(), 0.0);

        if(new_beta.histograms) {
            // Energy
            observables.energy_distribution = BuildHistogram(energy);
            // Overlap
            std::vector<std::pair<int, int>> overlap_pairs = BuildReplicaPairs();
            std::vector<double> overlap_samples(overlap_pairs.size());

            std::transform(overlap_pairs.begin(), overlap_pairs.end(), overlap_samples.begin(),
                [&](std::pair<int, int> p){return Overlap(replicas_[p.first], replicas_[p.second]);});
            observables.overlap = BuildHistogram(overlap_samples);
            // Link Overlap
            std::transform(overlap_pairs.begin(), overlap_pairs.end(), overlap_samples.begin(),
                [&](std::pair<int, int> p){return LinkOverlap(replicas_[p.first], replicas_[p.second]);});
            observables.link_overlap = BuildHistogram(overlap_samples);
        }

        observables.seed = rng_.GetSeed();
        results.push_back(observables);
    }
    return results;
}

bool PopulationAnnealing::AcceptedMove(double delta_energy) {
    if(delta_energy < 0.0) {
        return true;
    }
    // Get probability exponent and test probability
    double acceptance_prob_exp = -delta_energy*beta_;
    double test = rng_.Probability();

    // Compute bound on log of test number
    int lower_index = std::floor(test * (log_lookup_table_.size() - 1));
    double test_log_lower_bound = log_lookup_table_.at(lower_index);
    double test_log_upper_bound = log_lookup_table_.at(lower_index+1);

    // return acceptance_prob_exp > log(test);
    if(test_log_upper_bound < acceptance_prob_exp) {
        return true;
    }else if(test_log_lower_bound > acceptance_prob_exp) {
        return false;
    }
    // Compute exp if LUT can't resolve it
    return std::exp(acceptance_prob_exp) > test;
}

double PopulationAnnealing::Resample(double new_beta) {
    std::vector<StateVector> resampled_replicas;
    std::vector<int> resampled_families;
    resampled_replicas.reserve(replicas_.size());
    resampled_families.reserve(replicas_.size());
    
    Eigen::VectorXd weighting(replicas_.size());
    for(std::size_t k = 0; k < replicas_.size(); ++k) {
        weighting(k) = std::exp(-(new_beta-beta_) * Hamiltonian(replicas_[k]));
    }
    
    double normalize = average_population_ / weighting.sum();
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
    return normalize;
}
