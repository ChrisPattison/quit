#include "simd_xsadd.hpp"

namespace propane {
XSadd::XSadd(std::uint32_t seed) {
    // init RNG
    for(std::uint32_t i = 0; i < kvector_size; ++i) {
        state_[0][i] = seed ^ i;
        state_[1][i] = 0;
        state_[2][i] = 0;
        state_[3][i] = 0;
        
        state_[0][i] = state_[0][i] == 0 ? ~state_[0][i] : state_[0][i];
    }

    for(std::uint32_t i = 0; i < kvector_size; ++i) {
        for(std::uint32_t j = 1; j < 8; ++j) {
            state_[j & 3][i] ^= j + 1812433253U
                * (state_[(j-1) & 3][i] ^ (state_[(j-1) & 3][i] >> 30));
        }
    }

    for(std::uint32_t i = 0; i <= 8; ++i) {
        Advance();
    }
}
}