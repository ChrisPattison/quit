#include "simd_xsadd.hpp"
#include "XSadd/xsadd.h"
#include <vector>
#include "gtest/gtest.h"
#include <chrono>

namespace {

class XSaddTest : public ::testing::Test {
public:
    static constexpr std::uint32_t seed = 42U;

    std::vector<xsadd_t> xsadd_states_;
    propane::XSadd rng_;

    virtual void SetUp() {
        rng_.Seed(seed);
        xsadd_states_.resize(8);
        for(int i = 0; i < propane::XSadd::kvector_size; ++i) {
            xsadd_init(&xsadd_states_[i], seed ^ i);
        }
    }
};

TEST_F(XSaddTest, Benchmark) {
    const int krng_calls = 100000000;

    float sum = 0;
    auto time_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < krng_calls; ++i) {
        // Avoids optimization
        sum += xsadd_float(&xsadd_states_[0]);
    }
    auto serial_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - time_start).count();

    ASSERT_LE(sum, krng_calls);
    sum = 0;
    time_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < krng_calls; ++i) {
        sum += rng_.Uniform();
    }
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - time_start).count();

    EXPECT_GT(static_cast<double>(serial_time)/simd_time, 2.0);
}

TEST_F(XSaddTest, MatchesReferenceImplementation) {
    for(int k = 0; k < 10; ++k) {
        for(int i = 8; i > 0; ) {
            ASSERT_FLOAT_EQ(xsadd_float(&xsadd_states_.at(--i)), rng_.Uniform());
        }
    }
}
}
