/* Copyright (c) 2018 C. Pattison
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
#include <cmath>
#include "constant.hpp"

namespace propane { namespace util {
/** class for evaluating trig functions with discrete arguments
 */
class DiscreteSine {
public:
    std::array<double, propane::constant::ksine_lut_size> sin_table_;
    std::array<double, propane::constant::ksine_lut_size> cos_table_;

    struct xy_value {
        double sin;
        double cos;
    };

    DiscreteSine() {
        for(std::size_t i = 0; i < sin_table_.size(); ++i) {
            auto angle = (i * propane::constant::pi) / sin_table_.size();
            sin_table_[i] = std::sin(angle);
            cos_table_[i] = std::cos(angle);
        }
    }
    /** X and Y values for a fraction of the unit circle
     * operator() with angle/2pi
     */
    inline xy_value Unit(double uniform) const {
        bool parity = uniform < 0;
        // [0, 0.5) -> [0, N)
        std::size_t unrestricted_index = std::abs(uniform * 2 * sin_table_.size());
        auto index = unrestricted_index % sin_table_.size();
        bool half_period = ((unrestricted_index / sin_table_.size()) / 2 == 1);

        xy_value value = {sin_table_[index], cos_table_[index]};
        if (half_period) {
            value.sin *= -1;
            value.cos *= -1;
        }

        if(parity) {
            value.sin *= -1;
        }
        return value;
    }
    /** Approximately evaluate the sine and cosine of an angle
     */
    inline xy_value operator()(double angle) const {
        return Unit(angle / (2 * propane::constant::pi));
    }
};
}}