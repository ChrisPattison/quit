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

        auto header = utilities::Split(buffer, kWhitespaceTokens, utilities::kStringSplitOptions_RemoveEmptyEntries);
        int coupler_multiplier = std::stoi(header[1]);
        std::vector<std::tuple<double, double, double>> values;

        do {
            std::getline(file, buffer);
            buffer = utilities::Split(buffer, kCommentToken)[0];
            tokens = utilities::Split(buffer, kWhitespaceTokens, utilities::kStringSplitOptions_RemoveEmptyEntries);
            if(tokens.size() >= 3) {
                values.push_back(std::make_tuple(std::stod(tokens[0]), std::stod(tokens[1]), std::stod(tokens[2]) / coupler_multiplier));
            }
        } while(!file.eof());
        
        model.Resize(stoi(header[0]), values.size());
        for(auto& v : values) {
            model.AddEdge(std::get<0>(v), std::get<1>(v), std::get<2>(v));
            model.AddEdge(std::get<1>(v), std::get<0>(v), std::get<2>(v));
        }
        utilities::Check(model.IsConsistent(), "Missing edge somewhere.");
        utilities::Check(model.Adjacent().size() > 0, "No Elements.");
    } catch(std::exception& e) {
        utilities::Check(false, "Parsing failed.");
    }
}
}