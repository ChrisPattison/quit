#include "parse.hpp"
#include "graph.hpp"
#include "parallel_population_annealing.hpp"
#include "post.hpp"
#include "types.hpp"
#include "Eigen/Dense"
#include "utilities.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    Graph model;

    utilities::Check(argc >= 2, "Input File Expected");
    auto file = std::ifstream(argv[1]);
    io::IjjParse(model, file);
    file.close();

    // initializate solver
    int R = 40000;
    constexpr int kWidth = 18;
    constexpr int kHeaderWidth = kWidth + 1;

    std::vector<PopulationAnnealing::Temperature> betalist(101);
    for(std::size_t i = 0; i < betalist.size(); ++i) {
        betalist[i].beta = i*5.0/(betalist.size()-1);
    }

    betalist.back().histograms = true;

    ParallelPopulationAnnealing population_annealing(model, betalist, R);

    auto results = population_annealing.Run();

    parallel::Mpi parallel;
    parallel.ExecRoot([&]() {
        std::cout << "# Massively Parallel Population Annealing Monte Carlo" << std::endl;
        std::cout << "# C. Pattison" << std::endl;
        std::cout << "# Built: " << __DATE__ << " " << __TIME__ << std::endl;
        std::cout << "# Input: " << argv[1] << std::endl;
        std::cout << "# Seed: " << std::hex << results.front().seed << std::dec << std::endl; 
        std::cout << "# R=" << R << std::endl;
        std::cout << "# N=" << parallel.size() << std::endl;
        std::cout << std::right << std::setw(kHeaderWidth)
            << "Beta," << std::setw(kHeaderWidth)
            << "MC_Walltime," << std::setw(kHeaderWidth) 
            << "Redist_Walltime," << std::setw(kHeaderWidth) 
            << "Obs_Walltime," << std::setw(kHeaderWidth) 
            << "<E>," << std::setw(kHeaderWidth) 
            << "Q," << std::setw(kHeaderWidth) 
            << "R," << std::setw(kHeaderWidth) 
            << "E_MIN," << std::setw(kHeaderWidth) 
            << "R_MIN," << std::setw(kHeaderWidth) 
            << "S," << std::setw(kHeaderWidth) 
            << "R_f_MAX" << std::setw(kHeaderWidth) << std::endl;
        for(auto r : results) {
            std::cout << std::setprecision(10) << std::scientific << std::setw(kWidth)
                << r.beta << "," << std::setw(kWidth) 
                << r.montecarlo_walltime << "," << std::setw(kWidth) 
                << r.redist_walltime << "," << std::setw(kWidth) 
                << r.observables_walltime << "," << std::setw(kWidth)
                << r.average_energy << "," << std::setw(kWidth) 
                << r.norm_factor << "," << std::setw(kWidth) 
                << r.population << "," << std::setw(kWidth) 
                << r.ground_energy << "," << std::setw(kWidth) 
                << r.grounded_replicas << "," << std::setw(kWidth) 
                << r.entropy << "," << std::setw(kWidth) 
                << r.max_family_size << std::setw(kWidth) << std::endl; 
        }
        std::cout << "%%%---%%%" << std::endl;
        for(auto r : results) {
            for(auto q : r.overlap) {
                std::cout << "|q, " 
                    << std::setprecision(4) << std::fixed << r.beta << ","
                    << std::setprecision(10) << std::scientific << std::setw(kWidth) << q.bin << ","
                    << std::setw(kWidth) << q.value << std::endl;
            }
            for(auto ql : r.link_overlap) {
                std::cout << "|ql, " 
                    << std::setprecision(4) << std::fixed << r.beta << ","
                    << std::setprecision(10) << std::scientific << std::setw(kWidth) << ql.bin << ","
                    << std::setw(kWidth) << ql.value << std::endl;
            }
            for(auto E : r.energy_distribution) {
                std::cout << "|E, "
                    << std::setprecision(4) << std::fixed << r.beta << ","
                    << std::setprecision(10) << std::scientific << std::setw(kWidth) << E.bin << ","
                    << std::setw(kWidth) << E.value << std::endl;
            }
        }
    });
    return EXIT_SUCCESS;
}
