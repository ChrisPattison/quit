#include "random_number_generator.hpp"
#include <random>
#include <thread>
#include <ctime>
#include <chrono>
#include <cassert>

RandomNumberGenerator::RandomNumberGenerator(std::uint64_t seed) {
    seed_ = seed;
    vslNewStream(&stream_, VSL_BRNG_SFMT19937, seed);
    vslSkipAheadStream(stream_, 2492*100000);
}

RandomNumberGenerator::RandomNumberGenerator() : RandomNumberGenerator(RandomSeed()) {};

RandomNumberGenerator::~RandomNumberGenerator() {
    vslDeleteStream(&stream_);
}

double RandomNumberGenerator::Probability() {
    double p;
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream_, 1, &p, 0.0, 1.0);
    return p;
}

std::uint64_t RandomNumberGenerator::RandomSeed() {
    return time(NULL) ^
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() ^
        std::hash<std::thread::id>()(std::this_thread::get_id()) ^
        rand() ^
        std::random_device()();
}

std::uint64_t RandomNumberGenerator::GetSeed() {
    return seed_;
}

int RandomNumberGenerator::Range(int N) {
    int v;
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream_, 1, &v, 0, N);
    return v;
}
