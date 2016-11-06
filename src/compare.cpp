#include <cstdint>
#include <cmath>

bool FuzzyULPCompare(const float& a, const float& b, const int err = 10) {
    if(std::signbit(a)!=std::signbit(b)) {
        return a == b;
    }
    auto ai = reinterpret_cast<const std::uint32_t&>(a);
    auto bi = reinterpret_cast<const std::uint32_t&>(b);

    return std::abs(ai - bi) < err;
}

bool FuzzyULPCompare(const double& a, const double& b, const int err = 10) {
    if(std::signbit(a)!=std::signbit(b)) {
        return a == b;
    }
    auto ai = reinterpret_cast<const std::uint64_t&>(a);
    auto bi = reinterpret_cast<const std::uint64_t&>(b);

    return std::abs(ai - bi) < err;
}

bool FuzzyEpsCompare(const float& a, const float& b, const float err = 1e-6) {
    return std::abs(a - b) < err;
}

bool FuzzyEpsCompare(const double& a, const double& b, const double err = 1e-15) {
    return std::abs(a - b) < err;
}

bool FuzzyCompare(const double& a, const double& b) {
    return FuzzyEpsCompare(a,b) || FuzzyULPCompare(a,b);
}

bool FuzzyCompare(const float& a, const float& b) {
    return FuzzyEpsCompare(a,b) || FuzzyULPCompare(a,b);
}