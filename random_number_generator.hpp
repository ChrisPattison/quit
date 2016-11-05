#pragma once
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <climits>
#include <mkl_vsl.h>

class RandomNumberGenerator {
    VSLStreamStatePtr stream_;
    std::uint64_t seed_;
public:
    RandomNumberGenerator(std::uint64_t seed);
    RandomNumberGenerator();
    ~RandomNumberGenerator();

    // Gets a random seed value
    std::uint64_t RandomSeed();
    // Returns the seed used for this instance
    std::uint64_t GetSeed();
    // random number in [0.0, 1.0]
    double Probability();
    // random integer in range [0,N)
    int Range(int N);
};
