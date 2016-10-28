#include "random_number_generator.hpp"
#include <thread>
#include <ctime>
#include <chrono>
#include <cassert>

RandomNumberGenerator::RandomNumberGenerator(int rank) {
    auto seed = time(NULL) ^
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() ^
        std::hash<std::thread::id>()(std::this_thread::get_id()) ^
        rand() ^
        rank ^
        std::random_device()();
    generator_ = std::mt19937_64(static_cast<std::mt19937_64::result_type>(seed));
    generator_.discard(generator_.state_size*100000);
    pool_size_ = 0;
    last_range_ = 0;
    last_short_range_ = 0;
}

RandomNumberGenerator::RandomNumberGenerator() : RandomNumberGenerator(0) {};

decltype(RandomNumberGenerator::generator_()) RandomNumberGenerator::operator()() {
    return generator_();
}

double RandomNumberGenerator::Probability() {
    return static_cast<double>(generator_())/generator_.max();
}

// See http://stackoverflow.com/q/2509679/
std::uint32_t RandomNumberGenerator::Range(std::uint32_t N) {
    assert(UINT32_MAX / 2 > N);
    if(N != last_range_) {
        last_range_ = N;
        discard_range_ = UINT32_MAX - UINT32_MAX % N;
    }
    std::uint32_t value;
    do {
        value = this->Get<std::uint32_t>();
    }while (value >= discard_range_);
    return value % N;
}

std::uint16_t RandomNumberGenerator::ShortRange(std::uint16_t N) {
    assert(UINT16_MAX / 2 > N);
    if(N != last_short_range_) {
        last_short_range_ = N;
        discard_short_range_ = UINT16_MAX - UINT16_MAX % N;
    }
    std::uint16_t value;
    do {
        value = this->Get<std::uint16_t>();
    }while (value >= discard_short_range_);
    return value % N;
}
