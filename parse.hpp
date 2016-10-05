#pragma once
#include "graph.hpp"
#include <iostream>

namespace io {
    
static constexpr std::size_t kBufferSize = 1024; // size of input buffer
static constexpr std::size_t kReservedVertices = 6; // average number of edges per vertex
static constexpr int kOutputCouplerCoeff = 1'000'000;
static constexpr char kOutputSeperator = ' ';
static constexpr char kCommentToken = {'#'};
static const std::vector<char> kWhitespaceTokens = {' ', '\t'};

void IjjParse(Graph& model, std::istream& file);

void IjjDump(Graph& model, std::ostream& stream);
}
