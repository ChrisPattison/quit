#include "monte_carlo_driver.hpp"
#include "utilities.hpp"
#include <sys/mman.h>
#include <fcntl.h>
#include <fstream>
#include <exception>
#include <string>
#include <cmath>
#include <limits>

MonteCarloDriver::MonteCarloDriver() {
    bus_ = -1;
    device_ = -1;
    // std::fstream devices_file("/proc/bus/pci/devices", std::ios::in);
    // Grab the first valid device for now.
    // This will be moved to a seperate class eventually
    // do {
    //     std::string line;
    //     std::getline(devices_file, line);
    //     std::vector<std::string> tokens = utilities::Split(line, ' ', utilities::kStringSplitOptions_RemoveEmptyEntries);
    //     if(tokens.size() >= 2) {
    //         int vendor = std::stoi(tokens[1].substr(0, 4), 0, 16);
    //         int id = std::stoi(tokens[1].substr(4, 4), 0, 16);
    //         if(vendor == kVendorID && id == kDeviceID) {
    //             bus_ = std::stoi(tokens[0].substr(0,2), 0, 16);
    //             device_ = std::stoi(tokens[0].substr(2,2), 0, 16);
    //             break;
    //         }
    //     }
    // }while(!(devices_file.rdstate & (std::ios::eofbit | std::ios::failbit)));

    // if(bus_ == -1 && device_ == -1) {
    //     throw std::exception("Valid Accelerator not found");
    // }

    // Testing hardcode
    bar0_disc_ = open("/sys/bus/pci/devices/0000:81:00.0/resource0", O_RDWR | O_SYNC);
    bar0_ = reinterpret_cast<volatile std::uint8_t*>(mmap(nullptr, kMemSize, PROT_READ | PROT_WRITE, MAP_SHARED, bar0_disc_, 0));
    if(!bar0_) {
        // throw std::exception("mmap failed.");
    }
}

MonteCarloDriver::~MonteCarloDriver() {
    close(bar0_disc_);
    munmap((void*)bar0_, kMemSize);
}

void MonteCarloDriver::SeedRng(std::uint32_t seed) {
    volatile std::uint32_t* seed_register = reinterpret_cast<volatile std::uint32_t*>(&bar0_[kSeedAddr]);
    // Force reseed
    *seed_register = ~seed;
    *seed_register = seed;
}


void MonteCarloDriver::Sweep(std::vector<std::uint32_t>* replica, std::uint32_t sweeps) {
    volatile std::uint32_t* spin_base = reinterpret_cast<volatile std::uint32_t*>(&bar0_[kSpinBase]);
    volatile std::uint32_t* sweep_register = reinterpret_cast<volatile std::uint32_t*>(&bar0_[kSweepAddr]);
    
    for(int i = 0; i < replica->size(); ++i) {
        spin_base[i] = replica->data()[i];
    }

    *sweep_register = sweeps;
    CompletionWait();

    for(int i = 0; i < replica->size(); ++i) {
        replica->data()[i] = spin_base[i];
    }
}

void MonteCarloDriver::CompletionWait() {
    volatile std::uint32_t* sweep_register = reinterpret_cast<volatile std::uint32_t*>(&bar0_[kSweepAddr]);
    do {
        ;
    }while(*sweep_register > 0);
}

void MonteCarloDriver::SetGraph(Graph& structure) {
    std::uint32_t spins = *reinterpret_cast<volatile std::uint32_t*>(&bar0_[kSpinAddr]);
    std::uint32_t dimension = *reinterpret_cast<volatile std::uint32_t*>(&bar0_[kDimensionAddr]);
    std::uint32_t length = std::rint(std::pow(spins, 1.0/dimension));

    assert(spins == structure.size());
    delta_energy_.resize(structure.size() * (2<<dimension));

    // Site
    for(int vertex = 0; vertex < structure.size(); ++vertex) {
        // Single LUT entry
        for(int entry = 0; entry < 2<<dimension; ++entry) {
            double sum = 0.0;
            // sum over each dimension
            for(int dim = 0; dim < dimension; ++dim) {
                int stride = 2>>dim;
                int neighbor_vertex = (vertex + stride + structure.size()) % structure.size();
                sum += structure.Adjacent().innerVector(vertex).coeff(neighbor_vertex) * (entry & (1<<(dim*2)) ? 1 : -1);

                neighbor_vertex = (vertex - stride + structure.size()) % structure.size();
                sum += structure.Adjacent().innerVector(vertex).coeff(neighbor_vertex) * (entry & (1<<(dim*2+1)) ? 1 : -1);
            }
            sum -= structure.Fields()(vertex);
            delta_energy_[vertex*(2<<dimension) + entry] = sum;
        }
    }
}

void MonteCarloDriver::SetProb(double beta) {
    volatile std::uint32_t* lut_register = reinterpret_cast<volatile std::uint32_t*>(&bar0_[kLutAddr]);
    // 31 bits set
    std::uint32_t fixed_mask = ~0>>1;

    for(int i = delta_energy_.size()-1; i >= 0; --i) {
        // 0 is -1
        // msb signifies non-unity transition

        // -1 -> 1 transition non-unity
        if(delta_energy_[i] < 0) {
            // 31 bit fixed point
            *lut_register = static_cast<std::uint32_t>(std::exp(-delta_energy_[i]*beta) * fixed_mask);
        }else {
        // 1 -> -1 non-unity
            *lut_register = static_cast<std::uint32_t>(std::exp(delta_energy_[i]*beta) * fixed_mask) | ~fixed_mask;
        }
    }
};