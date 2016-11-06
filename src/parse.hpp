#pragma once
#include "graph.hpp"
#include "population_annealing.hpp"
#include <iostream>
#include <string>
#include <cstdint>

namespace io 
{
struct Config {
    int population;
    std::uint64_t seed;
    std::vector<PopulationAnnealing::Schedule> schedule;
};
static constexpr std::size_t kBufferSize = 1024; // size of input buffer
static constexpr std::size_t kReservedVertices = 6; // average number of edges per vertex
static constexpr int kOutputCouplerCoeff = 1'000'000;
static constexpr char kOutputSeperator = ' ';
static constexpr char kCommentToken = {'#'};
static const std::vector<char> kWhitespaceTokens = {' ', '\t'};

Graph IjjParse(std::istream& file);

void IjjDump(Graph& model, std::ostream& stream);

Config ConfigParse(std::istream& file);
}
