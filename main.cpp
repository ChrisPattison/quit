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

    std::vector<double> betalist;
    betalist.resize(101);
    for(std::size_t i = 0; i < betalist.size(); ++i) {
        betalist[i] = i*5.0/(betalist.size()-1);
    }

    auto population_annealing = PopulationAnnealing(model, betalist, R);

    std::vector<PopulationAnnealing::Result> results;
    //solve
    population_annealing.Run(results);

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
