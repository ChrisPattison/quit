#include "parse.hpp"
#include <vector>
#include <algorithm>
#include <iterator>
#include <exception>
#include <tuple>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cfenv>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include "utilities.hpp"
#include "compare.hpp"

namespace io 
{
Graph IjjParse(std::istream& file) {
    Graph model;
    try {
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
            int i = std::get<0>(v);
            int j = std::get<1>(v);
            EdgeType coeff = std::get<2>(v);
            if(i != j) {
                model.AddEdge(i, j, coeff);
                model.AddEdge(j, i, coeff);
            }else {
                model.SetField(i, coeff);
            }
        }
        utilities::Check(model.IsConsistent(), "Missing edge somewhere.");
        utilities::Check(model.Adjacent().size() > 0, "No Elements.");
    } catch(std::exception& e) {
        utilities::Check(false, "Input parsing failed.");
    }
    return model;
}

void IjjDump(Graph& model, std::ostream& stream) {
    auto round_mode = std::fegetround();
    std::fesetround(FE_TONEAREST);
    stream << model.size() << kOutputSeperator << kOutputCouplerCoeff << std::endl;
    // Coefficients
    for(std::size_t k = 0; k < model.Adjacent().outerSize(); ++k) {
        for(Eigen::SparseTriangularView<Eigen::SparseMatrix<EdgeType>,Eigen::Upper>::InnerIterator 
            it(model.Adjacent().triangularView<Eigen::Upper>(), k); it; ++it) {
            double value = kOutputCouplerCoeff * it.value();
            stream << k << kOutputSeperator << it.index() << kOutputSeperator;
            if(FuzzyUlpCompare(value, std::lrint(value), 100)) {
                stream << std::lrint(value);
            }else {
                stream << value;
            }
            stream << std::endl;
        }
    }
    // Fields
    if(model.has_field()) {
        for(std::size_t k = 0; k < model.Fields().size(); ++k) {
            stream << k << kOutputSeperator << k << kOutputSeperator;
            double value = kOutputCouplerCoeff * model.Fields()[k];
            if(FuzzyUlpCompare(value, std::lrint(value), 100)) {
                stream << std::lrint(value);
            }else {
                stream << value;
            }
            stream << std::endl;
        }
    }
    std::fesetround(round_mode);
}


ParallelPopulationAnnealing::Config ConfigParse(std::istream& file) {
    ParallelPopulationAnnealing::Config config;
    try {
        boost::property_tree::ptree tree;
        boost::property_tree::read_json(file, tree);

        config.population = tree.get<int>("population");
        std::stringstream converter(tree.get<std::string>("seed", "0"));
        converter >> std::hex >> config.seed;
        for(auto& item : tree.get_child("schedule")) {
            config.schedule.emplace_back();
            config.schedule.back().beta = item.second.get<double>("beta");
            config.schedule.back().sweeps = item.second.get("sweeps", 10);
            config.schedule.back().overlap_dist = item.second.get("overlap_hist", false);
            config.schedule.back().energy_dist = item.second.get("energy_hist", false);
        }
    } catch(std::exception& e) {
        utilities::Check(false, "Config parsing failed.");
    }
    std::sort(config.schedule.begin(), config.schedule.end(), [](const auto& left, const auto& right) {return left.beta < right.beta;});
    return config;
}
}