#pragma once
#include "graph.hpp"

namespace io {
    
static constexpr std::size_t kBufferSize = 1024; // size of input buffer
static constexpr std::size_t kReservedVertices = 6; // average number of edges per vertex
static constexpr char kCommentToken = {'#'};
static const std::vector<char> kWhitespaceTokens = {' ', '\t'};

void IjjParse(Graph& model, int& argc, char**& argv);
}
