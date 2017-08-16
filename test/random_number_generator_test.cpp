#include "simd_xsadd.hpp"
#include "XSadd/xsadd.h"
#include <vector>
#include "gtest/gtest.h"

namespace {
    TEST(XSadd, MatchesReferenceImplementation) {
        std::uint32_t seed = 42U;
        std::vector<xsadd_t> xsadd_states;
        xsadd_states.resize(8);
        for(int i = 0; i < propane::XSadd::kvector_size; ++i) {
            xsadd_init(&xsadd_states[i], seed ^ i);
        }

        propane::XSadd rng(seed);

        for(int k = 0; k < 4; ++k) {
            for(int i = 8; i > 0; --i) {
                ASSERT_FLOAT_EQ(xsadd_float(&xsadd_states.at(i-1)), rng.Uniform());
            }
        }
    }
}
