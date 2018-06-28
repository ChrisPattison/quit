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
#include <array>
#include <limits>
#include <cmath>
#include "constant.hpp"

namespace propane { namespace util {
/** Utility to quickly obtain a bound for the natural logarithm of a value
 */
class LogLookup {
    std::array<double, propane::constant::klog_lut_size> lookup_table_;
public:
    struct Bound {
        double upper;
        double lower;
    };
/** Allocates and fills lookup table
 */
    LogLookup();
/** Returns the upper and lower bound of the natural log of a number in [0,1]
 * Returns +/- inf if out of bounds
 */
    inline Bound GetBound(double value) const {
    unsigned int lower_index = static_cast<unsigned int>(std::floor(value * (lookup_table_.size() - 1)));
    // Bounds checking
    if(lower_index < 0 || lower_index >= lookup_table_.size() - 1) {
        return {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    }
    return {lookup_table_[lower_index+1], lookup_table_[lower_index]};
    }

    inline Bound operator()(double value) const {
        return GetBound(value);
    }
};
}}