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

#pragma once
#include <cstdint>
#include "immintrin.h"
// TODO: make sure metropolis assignment uses conditional assignment
namespace propane {
/** SIMD implementation of XSadd
 */
class XSadd {
public:
    static constexpr int kvector_size = 8;
    static constexpr int koutput_batches = 1;
private:
    static constexpr float kfloat_norm = (1.0f / 16777216.0f);
    typedef std::uint32_t vector_t __attribute__ ((vector_size (4*kvector_size)));
    typedef float float_vector_t __attribute__ ((vector_size (4*kvector_size)));

    __m256i __attribute__((aligned(32))) state_[4];
    float __attribute__((aligned(32))) output_[koutput_batches*kvector_size];
    std::uint32_t count_;

    inline void Advance();
public:
    XSadd();

    XSadd(std::uint32_t seed);

    void Seed(std::uint32_t seed);

    float Uniform();

    float Probability() { return Uniform(); }

    int Range(int N);
};
}

#include "simd_xsadd.inl"