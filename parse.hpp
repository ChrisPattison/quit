#pragma once
#include "graph.hpp"

const std::size_t kBufferSize = 1024; // size of input buffer
const std::size_t kReservedVertices = 6; // average number of edges per vertex
const std::vector<char> kWhitespaceTokens = {' ', '\t'};
const char kCommentToken = {'#'};

void IjjParse(Graph& model, int& argc, char**& argv);
void NativeParse(Graph& model, int& argc, char**& argv);