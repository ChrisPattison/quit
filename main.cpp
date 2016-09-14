#include "parse.hpp"
#include "graph.hpp"
#include "population_annealing.hpp"
#include "post.hpp"
#include "types.hpp"
#include "Eigen/Dense"
#include <iostream>

int main(int argc, char** argv) {
    Graph model;

    IjjParse(model, argc, argv);
    // NativeParse(model, argc, argv);

    std::cout.precision(6);
    std::cout.width(16);

    // initializate solver
    int R = 50000;
    // int R = 250000;

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

    auto population_annealing = PopulationAnnealing(model, betalist, R);

    std::vector<PopulationAnnealing::Result> results;
    //solve
    population_annealing.Run(results);

    // for (auto r : results) {
    //     std::cout << r.beta << std::endl;
    //     for(auto pd : r.overlap) {
    //         std::cout << pd.bin << ",\t" << pd.density << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    // for(std::size_t i = 1; i < results.size()-1; ++i) {
    //     std::cout << results[i].beta << ",\t" << results[i].energy.sum() / results[i].energy.size() << std::endl;
    // }
    // double Fsum = 0.0;
    // for(std::size_t i = 0; i < results.size()-1; ++i) {
    //     Fsum += std::log((results[i].energy.array() * -(betalist[i] - betalist[i-1])).exp().sum() / results[i].population);
    // }
    // for(std::size_t i = results.size()-1; i > 0; --i) {
    //     double Q = (results[i].energy.array() * -(betalist[i] - betalist[i-1])).exp().sum() / results[i].population;
    //     Fsum -= std::log(Q);
    //     double F = (Fsum + std::log(std::pow(2.0, model.Size()))) / -betalist[i];
    //     std::cout << betalist[i] << ",\t" << F << std::endl;
    // }
    return EXIT_SUCCESS;
}
