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
#include <numeric>
#include <Eigen/Dense>

namespace propane {
    
using EdgeType = double;
const double kEpsilon = 1e-13;

struct VertexType : std::array<double, 2> {
    VertexType() {
        (*this)[0] = 0;
        (*this)[1] = 0;
    }

    VertexType(int a) {
        (*this)[0] = a;
        (*this)[1] = 0;
    }

    VertexType(double a) {
        (*this)[0] = a;
        (*this)[1] = 0;
    }

    VertexType(double a, double b) {
        (*this)[0] = a;
        (*this)[1] = b;
    }

    inline double operator*(propane::VertexType b) const {
        return (*this)[0] * b[0] + (*this)[1] * b[1];
    }

    inline propane::VertexType operator/(double b) const {
         return (*this) * 1./b;
    }

    inline propane::VertexType operator*(int b) const {
        return {(*this)[0] * b, (*this)[1] * b};
    }

    inline propane::VertexType operator*(double b) const {
        return {(*this)[0] * b, (*this)[1] * b};
    }

    inline propane::VertexType operator+(const propane::VertexType b) const {
        return {(*this)[0] + b[0], (*this)[1] + b[1]};
    }

    inline propane::VertexType operator-(const propane::VertexType b) const {
        return {(*this)[0] - b[0], (*this)[1] - b[1]};
    }

    inline propane::VertexType operator*=(double b) {
        (*this) = (*this) * b;
        return *this;
    }
};
}

// inline double operator*(propane::VertexType a, propane::VertexType b) {
//     return a[0] * b[0] + a[1] * b[1];
// }


inline propane::VertexType operator*(const int& a, const propane::VertexType& b) {
    return b * a;
}

inline propane::VertexType operator*(const double& a, const propane::VertexType& b) {
    return b * a;
}

namespace Eigen {
template<> struct NumTraits<propane::VertexType> : NumTraits<double> {
};
}
    
