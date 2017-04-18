#pragma once
#include "population_annealing.hpp"
#include "types.hpp"
#include "graph.hpp"
#include "monte_carlo_driver.hpp"
#include <Eigen/Dense>

namespace propane
{
/** Accelerated Implementation of Population Annealing Monte Carlo.
 */
class FpgaPopulationAnnealing : public PopulationAnnealing{
    MonteCarloDriver driver_;
public:
protected:

using SpinPack = std::array<std::uint64_t, 4>;
using StateVectorPack = Eigen::Matrix<SpinPack, Eigen::Dynamic, 1>;

/** Carries out moves monte carlo sweeps of all replicas on the accelerator.
 */
    void Sweep(int moves);
/** Returns true if a move may be made that reduces the total energy.
 */
    double Resample(double new_beta, double population_fraction);
/** Returns new population size
 * Uses a logistic curve with parameters given in input file
 * This probably will be removed in the future with preference given 
 * to the current method of specifying beta schedules (one per temperature)
 */
public:

    FpgaPopulationAnnealing() = delete;
/** Intializes solver.
 * The inputs will be replaced by a struct in the future.
 * schedule specifies the annealing schedule, sweep counts, and histogram generation at each step.
 * seed may be zero in which case one will be generated.
 */
    FpgaPopulationAnnealing(Graph& structure, Config schedule);
/** Run solver and return results.
 */
};
}