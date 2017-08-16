#pragma once
#include <cstdint>

namespace propane {
class XSadd {
public:
    static constexpr int kvector_size = 8;
private:
    static constexpr float kfloat_norm = (1.0f / 16777216.0f);
    typedef std::uint32_t vector_t __attribute__ ((vector_size (4*kvector_size)));
    typedef float float_vector_t __attribute__ ((vector_size (4*kvector_size)));

    vector_t __attribute__((aligned(32))) state_[4];
    float_vector_t __attribute__((aligned(32))) output_;
    std::uint32_t count_;

    inline void Advance();
public:
    XSadd(std::uint32_t seed);

    float Uniform();
};

inline void XSadd::Advance() {
    auto temp = state_[0];
    temp ^= temp << 15;
    temp ^= temp >> 18;
    temp ^= state_[3] << 11;
    
    state_[0] = state_[1];
    state_[1] = state_[2];
    state_[2] = state_[3];
    state_[3] = temp;

    // This cast should be vectorized
    for(int i = 0; i < kvector_size; ++i) {
        output_[i] = static_cast<float>((state_[3][i] + state_[2][i]) >> 8) * kfloat_norm;
    }

    // output_ = (float_vector_t)((state_[3] + state_[2]) >> 8) * (float_vector_t){1.,1.,1.,1.}*kfloat_norm;
    count_ = kvector_size;
}

inline float XSadd::Uniform() {
    float output = output_[--count_];
    if(count_ == 0) {
        Advance();
    }
    return output;
}
}