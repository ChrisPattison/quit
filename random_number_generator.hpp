#pragma once
#include <random>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <climits>

class RandomNumberGenerator {
    std::mt19937_64 generator_;
    decltype(generator_()) pool_;
    unsigned char pool_size_;
    std::uint16_t last_range_;
    std::uint16_t discard_range_;
public:
    RandomNumberGenerator(std::size_t rank);
    RandomNumberGenerator();
    // random number
    template<typename T> std::enable_if_t<std::is_integral<T>::value, T> Get();
    //pass through () operator
    decltype(generator_()) operator()();
    // random number in [0.0, 1.0]
    double Probability();
    // random integer in range [0,N)
    std::uint16_t Range(std::uint16_t N);
};

template<typename T> std::enable_if_t<std::is_integral<T>::value, T> RandomNumberGenerator::Get() {
    // static_assert(sizeof(T) <= sizeof(decltype(pool_)));
    // this should pass through values for sizeof(T) = sizeof(decltype(pool_)) in the future

    const unsigned char T_bits = std::is_same<T, bool>::value ? 1 : sizeof(T) * CHAR_BIT;

    if(pool_size_ < T_bits) {
        pool_size_ = sizeof(decltype(pool_)) * CHAR_BIT;
        pool_ = generator_();
    }
    decltype(pool_) mask = ~(~0 << T_bits);
    decltype(pool_) bits = ~(~0 << T_bits) & pool_;
    pool_ >>= T_bits;
    pool_size_ -= T_bits;
    return *reinterpret_cast<T*>(&bits);
}
