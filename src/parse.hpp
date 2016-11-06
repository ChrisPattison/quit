#pragma once
#include "graph.hpp"
#include "population_annealing.hpp"
#include <iostream>
#include <string>
#include <cstdint>

namespace io 
{
/** Solver configuration options
 */
struct Config {
    int population;
    std::uint64_t seed;
    std::vector<PopulationAnnealing::Schedule> schedule;
};
static constexpr std::size_t kBufferSize = 1024; // size of input buffer
static constexpr std::size_t kReservedVertices = 6; // average number of edges per vertex
static constexpr int kOutputCouplerCoeff = 1000'000'000;
static constexpr char kOutputSeperator = ' ';
static constexpr char kCommentToken = {'#'};
static const std::vector<char> kWhitespaceTokens = {' ', '\t'};

/** Reads the bonds from file and constructs the Graph accordingly.
 * Expects a header with the number of spins and scaling factor.
 * Bond format is
 * i j J_ij
 */
Graph IjjParse(std::istream& file);
/** Opposite of IjjParse: Dumps bonds to file.
 */
void IjjDump(Graph& model, std::ostream& stream);
/** Parses configuration file and populates a Config.
 * Uses boost::property_tree to read the JSON.
 */
Config ConfigParse(std::istream& file);
}
