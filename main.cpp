#include "parse.hpp"
#include "graph.hpp"
#include "parallel_population_annealing.hpp"
#include "post.hpp"
#include "types.hpp"
#include "Eigen/Dense"
#include <iostream>

int main(int argc, char** argv) {
    Graph model;

    io::IjjParse(model, argc, argv);

    std::cout.precision(6);
    std::cout.width(16);

    // initializate solver
    int R = 1'000'000;

    std::vector<double> betalist = {
        0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 
        1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 
        1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2, 
        2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.380, 2.45, 2.5, 
        2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3, 
        3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 
        3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95, 4, 
        4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45, 4.5, 
        4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5,};
    // std::vector<double> betalist(201);
    // for(std::size_t i = 0; i < betalist.size(); ++i) {
    //     betalist[i] = i*10.0/(betalist.size()-1);
    // }

    ParallelPopulationAnnealing population_annealing(model, betalist, R);

    auto results = population_annealing.Run();

    Parallel parallel;
    parallel.ExecRoot([&]() {
        std::cout << "# R=" << R << " N=" << parallel.size() << std::endl;
        std::cout << "Beta,\tMC_Walltime,\tRedist_Walltime,\tObs_Walltime,\t<E>,\tR,\tE_MIN,\tR_MIN,\tR_MIN/R,\tS,\tR/e^S" << std::endl;
        for(auto r : results) {
            std::cout 
                << r.beta << ",\t" 
                << r.montecarlo_walltime << ",\t" 
                << r.redist_walltime << ",\t" 
                << r.observables_walltime << ",\t"
                << r.average_energy << ",\t" 
                << r.population << ",\t" 
                << r.ground_energy << ",\t" 
                << r.grounded_replicas << ",\t" 
                << static_cast<double>(r.grounded_replicas)/r.population << ",\t" 
                << r.entropy << ",\t" 
                << r.population/std::exp(r.entropy) << std::endl; 
        }
        std::cout << std::endl << std::endl << "q" << std::endl;
        for(auto q : results.back().overlap) {
            std::cout << q.bin << ",\t" << q.value << std::endl;
        }
        std::cout << std::endl << std::endl << "ql" << std::endl;
        for(auto q : results.back().link_overlap) {
            std::cout << q.bin << ",\t" << q.value << std::endl;
        }
    });
    return EXIT_SUCCESS;
}
