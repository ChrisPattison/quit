/* Copyright (c) 2016 C. Pattison
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
#include "parse.hpp"
#include "graph.hpp"
#include "parallel_tempering.hpp"
#include "types.hpp"
#include "string_util.hpp"
#include "version.hpp"
#include "output.hpp"
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <map>

/** Read model and config for regular PT
 */
void SinglePtPre(std::string& config_path, std::string& bond_path, propane::Graph* model, propane::ParallelTempering::Config* config, double planted_energy) {
    auto file = std::ifstream(config_path);
    propane::io::PtConfigParse(file, config, planted_energy);
    file.close();

    file = std::ifstream(bond_path);
    *model = propane::io::IjjParse(file);
    file.close();

    propane::io::Header(*model, config_path, bond_path);
}

void SinglePtPost(std::vector<propane::ParallelTempering::Result>& results, propane::Graph& model) {
    propane::io::PtColumnNames();
    std::cout << std::endl;
    for(auto& r : results) {
        propane::io::PtResults(r);
        std::cout << std::endl;
    }
    propane::io::IjjDump(model, std::cout);
    propane::io::PtHistograms(results);
}

void SinglePt(std::string config_path, std::string bond_path, double planted) {
    propane::Graph model;
    propane::ParallelTempering::Config config;
    SinglePtPre(config_path, bond_path, &model, &config, planted);
    
    propane::ParallelTempering parallel_tempering(model, config);
    auto results = parallel_tempering.Run();
    
    SinglePtPost(results, model);
}

enum ModeOption{
    kModeOptionPt
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
        ("version,v", "version number")
        ("bondfile", "file containing graph and couplers")
        ("mode,m", boost::program_options::value<std::string>()->default_value("pt"), "select run mode <1/pt>")
        ("plant,p", boost::program_options::value<double>()->default_value(std::numeric_limits<double>::quiet_NaN()), "planted/known ground state energy");
    boost::program_options::variables_map var_map;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv)
        .options(description).positional(positional_description).run(), var_map);
    boost::program_options::notify(var_map);

    // Print Help
    if(var_map.count("help") || argc == 1) {
        std::cout << "Quantum Inspired Tempering" << std::endl;
        std::cout << "C. Pattison" << std::endl << std::endl;
        std::cout << "Usage: " << argv[0] << " [options] <config> <bondfile>" << std::endl;
        std::cout << description << std::endl;
        return EXIT_SUCCESS;
    }

    if(var_map.count("version")) {
        std::cout << "QUantum Inspired Tempering" << std::endl;
        std::cout << "Branch: " << propane::version::kRefSpec << std::endl;
        std::cout << "Commit: " << std::string(propane::version::kCommitHash).substr(0, 8) << std::endl;
        std::cout << "Build:  " << propane::version::kBuildType << std::endl;
        std::cout << "Built:  " << propane::version::kBuildTime << std::endl;
        return EXIT_SUCCESS;
    }

    // Select PA implementation
    std::map<std::string, ModeOption> selector_map;
    selector_map.insert({"pt", kModeOptionPt});
    ModeOption selection;
    if(selector_map.count(var_map["mode"].as<std::string>()) == 0) {
        selection = kModeOptionPt;
    }else {
        selection = selector_map.at(var_map["mode"].as<std::string>());
    }

    double planted_energy = var_map["plant"].as<double>();

    switch(selection) {
        case kModeOptionPt : SinglePt(var_map["config"].as<std::string>(), var_map["bondfile"].as<std::string>(), planted_energy); break;
    }

    return EXIT_SUCCESS;
}
