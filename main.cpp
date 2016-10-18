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
    int R = 1'000'000;
    constexpr int kWidth = 18;
    constexpr int kHeaderWidth = kWidth + 1;

    // std::vector<double> hardbetalist = {
    //     0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
    //     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 
    //     1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 
    //     1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2, 
    //     2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.380, 2.45, 2.5, 
    //     2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3, 
    //     3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 
    //     3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95, 4, 
    //     4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45, 4.5, 
    //     4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5,};
    std::vector<PopulationAnnealing::Temperature> betalist(301);
    // for(auto b : hardbetalist) {
    //     betalist.push_back({b});
    // }
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
        std::cout << "# R=" << R << " N=" << parallel.size() << std::endl;
        std::cout << std::right << std::setw(kHeaderWidth)
            << "Beta," << std::setw(kHeaderWidth)
            << "MC_Walltime," << std::setw(kHeaderWidth) 
            << "Redist_Walltime," << std::setw(kHeaderWidth) 
            << "Obs_Walltime," << std::setw(kHeaderWidth) 
            << "<E>," << std::setw(kHeaderWidth) 
            << "R," << std::setw(kHeaderWidth) 
            << "E_MIN," << std::setw(kHeaderWidth) 
            << "R_MIN," << std::setw(kHeaderWidth) 
            << "R_MIN/R," << std::setw(kHeaderWidth) 
            << "S," << std::setw(kHeaderWidth) 
            << "R_f_MAX" << std::setw(kHeaderWidth) 
            << "R_N_MIN" << std::endl;
        for(auto r : results) {
            std::cout << std::setprecision(10) << std::scientific << std::setw(kWidth)
                << r.beta << "," << std::setw(kWidth) 
                << r.montecarlo_walltime << "," << std::setw(kWidth) 
                << r.redist_walltime << "," << std::setw(kWidth) 
                << r.observables_walltime << "," << std::setw(kWidth)
                << r.average_energy << "," << std::setw(kWidth) 
                << r.population << "," << std::setw(kWidth) 
                << r.ground_energy << "," << std::setw(kWidth) 
                << r.grounded_replicas << "," << std::setw(kWidth) 
                << static_cast<double>(r.grounded_replicas)/r.population << "," << std::setw(kWidth) 
                << r.entropy << "," << std::setw(kWidth) 
                << r.max_family_size << std::setw(kWidth) 
                << r.min_node_population << std::endl; 
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
        }
    });
    return EXIT_SUCCESS;
}
