/* Copyright (c) 2016 C. Pattison
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <cmath>

namespace propane {
inline void XSadd::Advance() {
    for(int i = 0; i < koutput_batches; ++i) {
        __m256i temp = state_[0];
        temp = _mm256_xor_si256(temp, _mm256_slli_epi32(temp, 15));
        temp = _mm256_xor_si256(temp, _mm256_srli_epi32(temp, 18));
        temp = _mm256_xor_si256(temp, _mm256_slli_epi32(state_[3], 11));
        
        state_[0] = state_[1];
        state_[1] = state_[2];
        state_[2] = state_[3];
        state_[3] = temp;

        __m256i bits = _mm256_srli_epi32(_mm256_add_epi32(state_[3], state_[2]), 8);
        _mm256_store_ps(output_ + i*kvector_size, _mm256_mul_ps(_mm256_cvtepi32_ps(bits), _mm256_set1_ps(kfloat_norm)));

    }
    count_ = kvector_size*koutput_batches;
}

inline float XSadd::Uniform() {
    float output = output_[--count_];
    if(count_ == 0) {
        Advance();
    }
    return output;
}

inline int XSadd::Range(int N) {
    return std::floor(Uniform() * N);
}

XSadd::XSadd() {
}

XSadd::XSadd(std::uint32_t seed) {
    Seed(seed);
}

void XSadd::Seed(std::uint32_t seed) {
    // init RNG
    state_[0] = _mm256_xor_si256(_mm256_set_epi32(7,6,5,4,3,2,1,0), _mm256_set1_epi32(seed));
    state_[1] = _mm256_setzero_si256();
    state_[2] = _mm256_setzero_si256();
    state_[3] = _mm256_setzero_si256();
    
    
    for(std::uint32_t j = 1; j < 8; ++j) {
        // state_[j & 3][i] ^= j + 1812433253U
        //     * (state_[(j-1) & 3][i] ^ (state_[(j-1) & 3][i] >> 30));
        state_[j & 3] = _mm256_xor_si256(state_[j & 3],
            _mm256_add_epi32(_mm256_set1_epi32(j),
                _mm256_mul_epu32(_mm256_set1_epi32(1812433253U), 
                    _mm256_xor_si256(state_[(j-1) & 3], _mm256_srli_epi32(state_[(j-1) & 3], 30)))));
    }

    for(std::uint32_t i = 0; i < 9; ++i) {
        Advance();
    }
}
}