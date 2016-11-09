#include "parse.hpp"
#include "graph.hpp"
#include "parallel_population_annealing.hpp"
#include "types.hpp"
#include "utilities.hpp"
#include "version.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    // Parse input and configuration files
    utilities::Check(argc >= 3, "Config and input files Expected");
    auto file = std::ifstream(argv[1]);
    auto config = io::ConfigParse(file);
    file.close();

    file = std::ifstream(argv[2]);
    Graph model = io::IjjParse(file);
    file.close();

    parallel::Mpi parallel;
    parallel.ExecRoot([&]() {
        std::cout << "# Massively Parallel Population Annealing Monte Carlo V" << version::kMajor << "." << version::kMinor << std::endl;
        std::cout << "# C. Pattison" << std::endl;
        std::cout << "# Branch: " << version::kRefSpec << std::endl;
        std::cout << "# Commit: " << version::kCommitHash << std::endl;
        std::cout << "# Built: " << version::kBuildTime << std::endl;
        std::cout << "# Config: " << argv[1] << std::endl;
        std::cout << "# Input: " << argv[2] << std::endl;
        std::cout << "# Cores: " << parallel.size() << std::endl;
        std::cout << "# Spins: " << model.size() << std::endl;
        std::cout << "# Population: " << config.population << std::endl;
    });
    // initializate solver and run
    ParallelPopulationAnnealing population_annealing(model, config.schedule, config.population, config.seed);
    auto results = population_annealing.Run();

    // Output formatting
    constexpr int kWidth = 18;
    constexpr int kHeaderWidth = kWidth + 1;
    const auto kMagicString = "%%%---%%%";

    parallel.ExecRoot([&]() {
        std::cout << "# Seed: " << std::hex << results.front().seed << std::dec << std::endl << std::endl; 
        std::cout << std::right << std::setw(kHeaderWidth)
            << "Beta" << std::setw(kHeaderWidth)
            << "Sweeps" << std::setw(kHeaderWidth)
            << "MC_Walltime" << std::setw(kHeaderWidth) 
            << "Redist_Walltime" << std::setw(kHeaderWidth) 
            << "Obs_Walltime" << std::setw(kHeaderWidth) 
            << "<E>" << std::setw(kHeaderWidth) 
            << "Q" << std::setw(kHeaderWidth) 
            << "R" << std::setw(kHeaderWidth) 
            << "E_MIN" << std::setw(kHeaderWidth) 
            << "R_MIN" << std::setw(kHeaderWidth) 
            << "S_f" << std::setw(kHeaderWidth) 
            << "R_f_MAX" << std::setw(kHeaderWidth)
            << "rho_t" << std::setw(kHeaderWidth) << std::endl;
        for(auto r : results) {
            std::cout << std::setprecision(10) << std::scientific << std::setw(kWidth)
                << r.beta << " " << std::setw(kWidth) 
                << r.sweeps << " " << std::setw(kWidth)
                << r.montecarlo_walltime << " " << std::setw(kWidth) 
                << r.redist_walltime << " " << std::setw(kWidth) 
                << r.observables_walltime << " " << std::setw(kWidth)
                << r.average_energy << " " << std::setw(kWidth) 
                << r.norm_factor << " " << std::setw(kWidth) 
                << r.population << " " << std::setw(kWidth) 
                << r.ground_energy << " " << std::setw(kWidth) 
                << r.grounded_replicas << " " << std::setw(kWidth) 
                << r.entropy << " " << std::setw(kWidth) 
                << r.max_family_size << " " << std::setw(kWidth) 
                << r.mean_square_family_size << std::endl; 
        }
        std::cout << std::endl << kMagicString << std::endl << "# Input" << std::endl;
        io::IjjDump(model, std::cout);
        std::cout << std::endl << kMagicString << std::endl << "# Histograms" << std::endl;
        for(auto r : results) {
            for(auto q : r.overlap) {
                std::cout << "|q, " 
                    << std::setprecision(4) << std::fixed << r.beta << " "
                    << std::setprecision(10) << std::scientific << std::setw(kWidth) << q.bin << " "
                    << std::setw(kWidth) << q.value << std::endl;
            }
            for(auto ql : r.link_overlap) {
                std::cout << "|ql, " 
                    << std::setprecision(4) << std::fixed << r.beta << " "
                    << std::setprecision(10) << std::scientific << std::setw(kWidth) << ql.bin << " "
                    << std::setw(kWidth) << ql.value << std::endl;
            }
            for(auto E : r.energy_distribution) {
                std::cout << "|E, "
                    << std::setprecision(4) << std::fixed << r.beta << " "
                    << std::setprecision(10) << std::scientific << std::setw(kWidth) << E.bin << " "
                    << std::setw(kWidth) << E.value << std::endl;
            }
        }
    });
    return EXIT_SUCCESS;
}
