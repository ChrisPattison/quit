#include "spin_vector_monte_carlo.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "gtest/gtest.h"

namespace {

class SingleSpinTester : public propane::SpinVectorMonteCarlo {
    StateVector state_;
public:
    SingleSpinTester() {
        structure_.Resize(2);
        structure_.AddEdge(0, 1, 0.0);
        structure_.AddEdge(1, 0, 0.0);
        structure_.SetField(0, 1.0);
        structure_.SetField(1, 1.0);
        state_.beta = 1.0;
        state_.gamma = 0.0;
        state_.resize(2);
        std::fill(state_.begin(), state_.end(), propane::VertexType(1.0, 0.0));
    }

    propane::VertexType SampleHeatbath() {
        HeatbathSweep(state_, 1);
        return state_[0];
    }

    propane::VertexType SampleMetropolis() {
        MetropolisSweep(state_, 20);
        return state_[0];
    }
};

class SpinVectorMonteCarloTest : public ::testing::Test  {
protected:
    SingleSpinTester tester_;
public:
    
    virtual void SetUp() {

    }
};

TEST_F(SpinVectorMonteCarloTest, HeatbathSampling) {

    // Compute metropolis moments
    std::vector<double> metropolis_samples;
    metropolis_samples.resize(10000);
    std::generate(metropolis_samples.begin(), metropolis_samples.end(), [&](){ return tester_.SampleMetropolis() * propane::FieldType(1.0, 0.0); });
    
    std::vector<double> metropolis_moments;
    metropolis_moments.resize(6);
    
    auto mean = std::accumulate(metropolis_samples.begin(), metropolis_samples.end(), 0.0)/metropolis_samples.size();
    for(auto n = 0U; n < metropolis_moments.size(); ++n) {
        metropolis_moments[n] = std::pow(mean, n) - std::accumulate(metropolis_samples.begin(), metropolis_samples.end(), 0.0, 
        [&](const auto& acc, auto const& v) {return acc + std::pow(v, n); })/metropolis_samples.size();
    }
    
    // Compute heatbath moments
    std::vector<double> heatbath_samples;
    heatbath_samples.resize(10000);
    std::generate(heatbath_samples.begin(), heatbath_samples.end(), [&](){ return tester_.SampleHeatbath() * propane::FieldType(1.0, 0.0); });
    
    std::vector<double> heatbath_moments;
    heatbath_moments.resize(6);
    
    mean = std::accumulate(heatbath_samples.begin(), heatbath_samples.end(), 0.0)/heatbath_samples.size();
    for(auto n = 0U; n < heatbath_moments.size(); ++n) {
        heatbath_moments[n] = std::pow(mean, n) - std::accumulate(heatbath_samples.begin(), heatbath_samples.end(), 0.0, 
        [&](const auto& acc, auto const& v) {return acc + std::pow(v, n); })/heatbath_samples.size();
    }
    
    // Compare moments
    EXPECT_FLOAT_EQ(heatbath_moments[0], metropolis_moments[0]);
    EXPECT_FLOAT_EQ(heatbath_moments[1], metropolis_moments[1]);
    EXPECT_FLOAT_EQ(heatbath_moments[2], metropolis_moments[2]);
    EXPECT_FLOAT_EQ(heatbath_moments[3], metropolis_moments[3]);
    EXPECT_FLOAT_EQ(heatbath_moments[4], metropolis_moments[4]);
    EXPECT_FLOAT_EQ(heatbath_moments[5], metropolis_moments[5]);
}
}