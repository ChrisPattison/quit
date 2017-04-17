#include "parse.hpp"
#include "graph.hpp"
#include "parallel_population_annealing.hpp"
#include "greedy_population_annealing.hpp"
#include "types.hpp"
#include "utilities.hpp"
#include "version.hpp"
#include "output.hpp"
#include <boost/program_options.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <map>

void MpiPa(std::string config_path, std::string bond_path) {
    auto file = std::ifstream(config_path);
    ParallelPopulationAnnealing::Config config;
    io::ConfigParse(file, &config);
    file.close();

    file = std::ifstream(bond_path);
    Graph model = io::IjjParse(file);
    file.close();

    parallel::Mpi parallel;
    parallel.ExecRoot([&]() {
        io::Header(model, config_path, bond_path);
        io::MpiHeader(parallel);
    });

    ParallelPopulationAnnealing population_annealing(model, config);
    auto results = population_annealing.Run();

    parallel.ExecRoot([&]() {
        io::ColumnNames();
        io::MpiColumnNames();
        std::cout << std::endl;
        for(auto& r : results) {
            io::Results(r);
            io::MpiResults(r);
            std::cout << std::endl;
        }
        std::vector<PopulationAnnealing::Result> basic_result(results.size());
        std::transform(results.begin(), results.end(), basic_result.begin(), [] (ParallelPopulationAnnealing::Result& r) 
            {return static_cast<PopulationAnnealing::Result>(r);});

        io::IjjDump(model, std::cout);
        io::Histograms(basic_result);
    });
}

/** Read model and config for regular PA
 */
void SinglePaPre(std::string& config_path, std::string& bond_path, Graph* model, PopulationAnnealing::Config* config) {
    auto file = std::ifstream(config_path);
    io::ConfigParse(file, config);
    file.close();

    file = std::ifstream(bond_path);
    *model = io::IjjParse(file);
    file.close();

    io::Header(*model, config_path, bond_path);
}

/** Output Data for regular PA
 */
void SinglePaPost(std::vector<PopulationAnnealing::Result>& results, Graph& model) {
    io::ColumnNames();
    std::cout << std::endl;
    for(auto& r : results) {
        io::Results(r);
        std::cout << std::endl;
    }
    io::IjjDump(model, std::cout);
    io::Histograms(results);
}

void SinglePa(std::string config_path, std::string bond_path) {
    Graph model;
    PopulationAnnealing::Config config;
    SinglePaPre(config_path, bond_path, &model, &config);

    PopulationAnnealing population_annealing(model, config);
    auto results = population_annealing.Run();

    SinglePaPost(results, model);
}

void GreedyPa(std::string config_path, std::string bond_path) {
    Graph model;
    GreedyPopulationAnnealing::Config config;
    SinglePaPre(config_path, bond_path, &model, &config);

    GreedyPopulationAnnealing population_annealing(model, config);
    auto results = population_annealing.Run();

    SinglePaPost(results, model);
}

void FpgaPa(std::string config_path, std::string bond_path) {
    Graph model;
    PopulationAnnealing::Config config;
    SinglePaPre(config_path, bond_path, &model, &config);

    FpgaPopulationAnnealing population_annealing(model, config);
    auto results = population_annealing.Run();

    SinglePaPost(results, model);
}

enum ModeOption{
    kModeOptionFpga,
    kModeOptionMpi,
    kModeOptionSingle,
    kModeOptionGreedy
};

int main(int argc, char** argv) {
    // Parse Arguments
    boost::program_options::options_description description("Options");
    boost::program_options::positional_options_description positional_description;
    positional_description.add("config", 1);
    positional_description.add("bondfile", 1);

    description.add_options()
        ("help,h", "help message")
        ("config", "configuration file")
        ("bondfile", "file containing graph and couplers")
        ("mode,m", boost::program_options::value<std::string>()->default_value("1"), "select run mode <1/mpi/fpga/greedy>");
    boost::program_options::variables_map var_map;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv)
        .options(description).positional(positional_description).run(), var_map);
    boost::program_options::notify(var_map);

    // Print Help
    if (var_map.count("help") || argc < 3) {
        std::cout << "Parallel Optimized Population Annealing V" << version::kMajor << "." << version::kMinor << std::endl;
        std::cout << "C. Pattison" << std::endl << std::endl;
        std::cout << "Usage: " << argv[0] << " [options] <config> <bondfile>" << std::endl;
        std::cout << description << std::endl;
        return EXIT_SUCCESS;
    }

    // Select PA implementation
    std::map<std::string, ModeOption> selector_map;
    selector_map.insert({"mpi", kModeOptionMpi});
    selector_map.insert({"fpga", kModeOptionFpga});
    selector_map.insert({"greedy", kModeOptionGreedy});
    ModeOption selection;
    if(selector_map.count(var_map["mode"].as<std::string>()) == 0) {
        selection = kModeOptionSingle;
    }else {
        selection = selector_map.at(var_map["mode"].as<std::string>());
    }

    switch(selection) {
        case kModeOptionMpi : MpiPa(var_map["config"].as<std::string>(), var_map["bondfile"].as<std::string>()); break;
        case kModeOptionFpga : FpgaPa(var_map["config"].as<std::string>(), var_map["bondfile"].as<std::string>()); break;
        case kModeOptionSingle : SinglePa(var_map["config"].as<std::string>(), var_map["bondfile"].as<std::string>()); break;
        case kModeOptionGreedy : GreedyPa(var_map["config"].as<std::string>(), var_map["bondfile"].as<std::string>()); break;
    }

    return EXIT_SUCCESS;
}
