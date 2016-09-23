#pragma once
#include "graph.hpp"

namespace io {
    
constexpr std::size_t kBufferSize = 1024; // size of input buffer
constexpr std::size_t kReservedVertices = 6; // average number of edges per vertex
constexpr char kCommentToken = {'#'};
const std::vector<char> kWhitespaceTokens = {' ', '\t'};

void IjjParse(Graph& model, int& argc, char**& argv);
}
