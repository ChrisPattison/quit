#include "parse.hpp"
#include <vector>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <exception>
#include <iostream>
#include <tuple>
#include "utilities.hpp"

void IjjParse(Graph& model, int& argc, char**& argv) {
    try {
        utilities::Check(argc >= 2, "Input File Expected");
        auto file = std::ifstream(argv[1]);
        std::string buffer(kBufferSize, '\0');
        std::vector<std::string> tokens;
        
        do {
            std::getline(file, buffer);
        } while(utilities::Split(buffer, kWhitespaceTokens, utilities::kStringSplitOptions_RemoveEmptyEntries)[0][0]=='#');

        int coupler_multiplier = std::stoi(utilities::Split(buffer, kWhitespaceTokens, utilities::kStringSplitOptions_RemoveEmptyEntries)[1]);
        std::vector<std::tuple<double, double, double>> values;

        do {
            std::getline(file, buffer);
            buffer = utilities::Split(buffer, kCommentToken)[0];
            tokens = utilities::Split(buffer, kWhitespaceTokens, utilities::kStringSplitOptions_RemoveEmptyEntries);
            if(tokens.size() >= 3) {
                values.push_back(std::make_tuple(std::stod(tokens[0]), std::stod(tokens[1]), std::stod(tokens[2]) / coupler_multiplier));
            }
        } while(!file.eof());
        
        model.Resize(values.size(), values.size() * kReservedVertices);
        for(auto& v : values) {
            model.AddEdge(std::get<0>(v), std::get<1>(v), std::get<2>(v));
            model.AddEdge(std::get<1>(v), std::get<0>(v), std::get<2>(v));
        }
        utilities::Check(model.IsSymmetric(), "Adjacency matrix not symmetric. Missing edge somewhere.");
        utilities::Check(model.Adjacent().size() > 0, "No Elements.");
    } catch(std::exception& e) {
        utilities::Check(false, "Parsing failed.");
    }
}

void NativeParse(Graph& model, int& argc, char**& argv) {
    try {
        utilities::Check(argc >= 3, "Expected graphsolve <graph file> <coupler>");

        // open input files and check consistency
        auto graph_file = std::ifstream(argv[1]);
        auto coupler_file = std::ifstream(argv[2]);

        std::string graph_buffer(kBufferSize, '\0');
        std::string coupler_buffer(kBufferSize, '\0');

        std::getline(graph_file, graph_buffer);
        std::getline(coupler_file, coupler_buffer);
        graph_buffer = utilities::Split(graph_buffer, '#')[0];
        coupler_buffer = utilities::Split(coupler_buffer, '#')[0];
        int count = std::stoi(graph_buffer);
        utilities::Check(count == std::stoi(coupler_buffer), "Vertex count between graph and coupler input files do not match");

        // initialize input graph
        model.Resize(count, count * kReservedVertices);

        std::vector<int> edges, values;
        std::vector<std::string> line_buffer;
        for(std::size_t vertex = 0; vertex < count; ++vertex) {
            edges.clear();
            values.clear();

            std::getline(graph_file, graph_buffer);
            graph_buffer = utilities::Split(graph_buffer, '#')[0];
            line_buffer = utilities::Split(graph_buffer, kWhitespaceTokens, utilities::kStringSplitOptions_RemoveEmptyEntries);
            std::for_each(line_buffer.begin(), line_buffer.end(), [&edges] (std::string& entry) {edges.push_back(std::stoi(entry));});

            std::getline(coupler_file, coupler_buffer);
            coupler_buffer = utilities::Split(coupler_buffer, '#')[0];
            line_buffer = utilities::Split(coupler_buffer, kWhitespaceTokens, utilities::kStringSplitOptions_RemoveEmptyEntries);
            model.SetField(vertex, std::stoi(line_buffer[0])); // grab field value 
            line_buffer.erase(line_buffer.begin());
            std::for_each(line_buffer.begin(), line_buffer.end(), [&values] (std::string& entry) {values.push_back(std::stoi(entry));});

            utilities::Check(values.size()==edges.size(), "Edge/Value entry length mismatch");

            auto length = edges.size();
            for(std::size_t k = 0; k < length; ++k) {
                model.AddEdge(vertex, edges.back(), values.back());
                edges.pop_back();
                values.pop_back();
            }
        }
        utilities::Check(model.IsSymmetric(), "Adjacency matrix not symmetric. Missing edge somewhere.");
        utilities::Check(model.Adjacent().size() > 0, "No Elements.");
    } catch(std::exception& e) {
        utilities::Check(false, "Parsing failed.");
    }
}