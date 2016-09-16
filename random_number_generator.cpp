#include "random_number_generator.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <cassert>

RandomNumberGenerator::RandomNumberGenerator(int rank) {
    generator_ = std::mt19937_64(static_cast<std::mt19937_64::result_type>(
        time(NULL) ^
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() ^
        std::hash<std::thread::id>()(std::this_thread::get_id()) ^
        rand() ^
        rank ^
        std::random_device()()));
    generator_.discard(generator_.state_size*100000);
    pool_size_ = 0;
    last_range_ = 0;
}

RandomNumberGenerator::RandomNumberGenerator() {
    generator_ = std::mt19937_64(static_cast<std::mt19937_64::result_type>(
        time(NULL) ^
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() ^
        std::hash<std::thread::id>()(std::this_thread::get_id()) ^
        rand() ^
        std::random_device()()));
    generator_.discard(generator_.state_size*100000);
    pool_size_ = 0;
    last_range_ = 0;
}

decltype(RandomNumberGenerator::generator_()) RandomNumberGenerator::operator()() {
    return generator_();
}

double RandomNumberGenerator::Probability() {
    return static_cast<double>(generator_())/generator_.max();
}

// See http://stackoverflow.com/q/2509679/
std::uint16_t RandomNumberGenerator::Range(std::uint16_t N) {
    assert(UINT16_MAX / 2 > N);
    if(N != last_range_) {
        last_range_ = N;
        discard_range_ = UINT16_MAX - UINT16_MAX % N;
    }
    std::uint16_t value;
    do {
        value = this->Get<std::uint16_t>();
    }while (value >= discard_range_);
    return value % N;
}
