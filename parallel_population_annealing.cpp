#include "parallel_population_annealing.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <limits>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <functional>

std::vector<std::pair<int, int>> ParallelPopulationAnnealing::BuildReplicaPairs() {
    std::vector<std::pair<int, int>> pairs;
    pairs.resize(replica_families_.size()/2);

    for(int k = 0; k < pairs.size(); ++k) {
        pairs[k] = {k, k + pairs.size()};
    }

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

void ParallelPopulationAnnealing::OverlapPmd(std::vector<Result::ProbabilityMass>& pmd) {
    std::vector<std::pair<int, int>> overlap_pairs = BuildReplicaPairs();
    std::vector<double> overlap(overlap_pairs.size());
    std::transform(overlap_pairs.begin(), overlap_pairs.end(), overlap.begin(),
        [&](std::pair<int, int> p){return Overlap(replicas_[p.first], replicas_[p.second]);});
    for(auto v : overlap) {
        auto it = std::lower_bound(pmd.begin(), pmd.end(), v, [&](const Result::ProbabilityMass& a, const double& b) {return kEpsilon < b - a.bin;});
        if(it==pmd.end() || v + kEpsilon <= it->bin) {
            pmd.insert(it, {v, 1.0});
        }else {
            it->mass++;
        }
    }
    for(auto& pd : pmd) {
        pd.mass /= overlap_pairs.size();
    }
}

std::vector<double> ParallelPopulationAnnealing::FamilyCount() {
    std::vector<double> count;
    count.reserve(replica_families_.size());
    auto i = replica_families_.begin();
    do {
        auto i_next = std::find_if(i, replica_families_.end(), [&](const int& v){return v != *i;});
        count.push_back(static_cast<double>(std::distance(i, i_next)) / replicas_.size());
        i = i_next;
    }while(i != replica_families_.end());
    return count;
}

ParallelPopulationAnnealing::ParallelPopulationAnnealing(Graph& structure, std::vector<double> betalist, int average_population) {
    rng_ = RandomNumberGenerator(parallel_.rank());
    betalist_ = betalist;
    beta_ = NAN;
    structure_ = structure;
    average_node_population_ = (average_population / parallel_.size());
    replicas_.resize(average_node_population_);
    replica_families_.resize(average_node_population_);

    for(auto& r : replicas_) {
        r = StateVector();
        r.resize(structure_.Fields().size());
    }
 }

double ParallelPopulationAnnealing::Hamiltonian(StateVector& replica) {
    double energy = 0.0;
    for(std::size_t k = 0; k < structure_.Adjacent().outerSize(); ++k) {
        for(Eigen::SparseTriangularView<Eigen::SparseMatrix<EdgeType>,Eigen::Upper>::InnerIterator 
            it(structure_.Adjacent().triangularView<Eigen::Upper>(), k); it; ++it) {
            energy += replica(k) * it.value() * replica(it.index());
        }
        // energy -= replica(k) * structure_.Fields()(k);
    }
    return energy;
}

double ParallelPopulationAnnealing::DeltaEnergy(StateVector& replica, int vertex) {
    double h = 0.0;
    for(Eigen::SparseMatrix<EdgeType>::InnerIterator it(structure_.Adjacent(), vertex); it; ++it) {
        h += it.value() * replica(it.index());
    }
    // h -= structure_.Fields()(vertex);
    return -2 * replica(vertex) * h;
}

void ParallelPopulationAnnealing::MonteCarloSweep(StateVector& replica, int sweeps) {
    for(std::size_t k = 0; k < sweeps; ++k) {
        for(std::size_t i = 0; i < replica.size(); ++i) {
            int vertex = rng_.Range(replica.size());
            double delta_energy = DeltaEnergy(replica, vertex);
            double acceptance_probability = AcceptanceProbability(delta_energy);
            
            //round-off isn't a concern here
            if(acceptance_probability==1.0 || acceptance_probability > rng_.Probability()) {
                replica(vertex) *= -1;
            }
        }
    }
}

bool ParallelPopulationAnnealing::IsLocalMinimum(StateVector& replica) {
    for(int k = 0; k < replica.size(); ++k) {
        if(DeltaEnergy(replica, k) < 0) {
            return false;
        }
    }
    return true;
}

ParallelPopulationAnnealing::StateVector ParallelPopulationAnnealing::Quench(const StateVector& replica) {
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

double ParallelPopulationAnnealing::Overlap(StateVector& alpha, StateVector& beta) {
    return static_cast<double>((alpha.array() * beta.array()).sum()) / structure_.size();
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

//TODO: compute the exponential only when necessary
double ParallelPopulationAnnealing::AcceptanceProbability(double delta_energy) const {
    return delta_energy < 0.0 ? 1.0 : std::exp(-delta_energy*beta_);
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
    
    double normalize = average_node_population_ * parallel_.size() / summed_weights;
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
