#pragma once
#include <vector>
#include <limits>

namespace propane 
{
class PopulationAnnealingBase {
    public:
/** Observables for a single step.
 */
    struct Result {
        struct Histogram {
            double bin;
            double value;
        };
        std::vector<Histogram> overlap;
        std::vector<Histogram> link_overlap;
        std::vector<Histogram> energy_distribution;
        double beta = std::numeric_limits<double>::quiet_NaN();
        int population = -1;
        double norm_factor = std::numeric_limits<double>::quiet_NaN();
        double average_energy = std::numeric_limits<double>::quiet_NaN();
        double average_squared_energy = std::numeric_limits<double>::quiet_NaN();
        double ground_energy = std::numeric_limits<double>::quiet_NaN();
        int grounded_replicas = 0;
        double entropy = std::numeric_limits<double>::quiet_NaN();
        std::uint64_t seed = 0;
        int sweeps = -1;
        double mean_square_family_size = std::numeric_limits<double>::quiet_NaN();
        double montecarlo_walltime = std::numeric_limits<double>::quiet_NaN();
    };
/** Parameters for a single step.
 */
    struct Schedule {
        double beta;
        int sweeps = 10;
        bool overlap_dist = false;
        bool energy_dist = false;
        bool compute_observables = false;
        double population_fraction = 1.0;
    };
/** Parameters for entire run.
 */
    struct Config {
        int population;
        std::uint64_t seed;
        std::vector<PopulationAnnealingBase::Schedule> schedule;
        bool solver_mode = false;
    };
};
}