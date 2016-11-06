#pragma once
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <climits>
#include <mkl_vsl.h>

/** Wrapper around the Math Kernel Library implementation of SFMT19937.
 */
class RandomNumberGenerator {
    VSLStreamStatePtr stream_;
    std::uint64_t seed_;
public:
/** Seed generator with given value.
 */
    RandomNumberGenerator(std::uint64_t seed);
/** Seed generator with value from RandomSeed.
 */
    RandomNumberGenerator();
    ~RandomNumberGenerator();

/** Returns a random value suitable for seeding.
 */
    std::uint64_t RandomSeed();
/** Returns the value used to seed this instance.
 */
    std::uint64_t GetSeed();
/** Returns a double uniformly distributed in [0,1).
 */
    double Probability();
/** Returns an integer uniformly distributed in [0,N).
 */
    int Range(int N);
};
