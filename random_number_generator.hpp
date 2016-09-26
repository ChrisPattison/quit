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
    std::uint16_t last_short_range_;
    std::uint16_t discard_short_range_;
    std::uint32_t last_range_;
    std::uint32_t discard_range_;
public:
    RandomNumberGenerator(int rank);
    RandomNumberGenerator();
    // random number
    template<typename T> auto Get() ->
    std::enable_if_t<std::is_integral<T>::value, std::enable_if_t<sizeof(T) < sizeof(decltype(pool_)), T>>;
    template<typename T> auto Get() ->
    std::enable_if_t<std::is_integral<T>::value, std::enable_if_t<sizeof(T) == sizeof(decltype(pool_)), T>>;
    //pass through () operator
    decltype(generator_()) operator()();
    // random number in [0.0, 1.0]
    double Probability();
    // random integer in range [0,N)
    std::uint16_t ShortRange(std::uint16_t N);
    std::uint32_t Range(std::uint32_t N);
};

template<typename T> auto RandomNumberGenerator::Get() ->
std::enable_if_t<std::is_integral<T>::value, std::enable_if_t<sizeof(T) < sizeof(decltype(pool_)), T>> {
    const unsigned char T_bits = std::is_same<T, bool>::value ? 1 : sizeof(T) * CHAR_BIT;

    if(pool_size_ < T_bits) {
        pool_size_ = sizeof(decltype(pool_)) * CHAR_BIT;
        pool_ = generator_();
    }
    decltype(pool_) bits = ~(~static_cast<decltype(pool_)>(0) << T_bits) & pool_;
    pool_ >>= T_bits;
    pool_size_ -= T_bits;
    return *reinterpret_cast<T*>(&bits);
}

template<typename T> auto RandomNumberGenerator::Get() ->
std::enable_if_t<std::is_integral<T>::value, std::enable_if_t<sizeof(T) == sizeof(decltype(pool_)), T>> {
    auto value = generator_();
    return *reinterpret_cast<T*>(&value);
}