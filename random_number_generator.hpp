#pragma once
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <climits>
#include <mkl_vsl.h>

class RandomNumberGenerator {
    VSLStreamStatePtr stream_;
public:
    RandomNumberGenerator(std::uint64_t seed);
    RandomNumberGenerator();
    ~RandomNumberGenerator();

    // Gets a seed value
    std::uint64_t RandomSeed();
    // random number in [0.0, 1.0]
    double Probability();
    // random integer in range [0,N)
    int Range(int N);
};
