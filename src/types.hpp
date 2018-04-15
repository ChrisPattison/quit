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
#include <Eigen/Dense>

namespace propane {

/** struct that represents a vector in R^2
 */
struct FieldType : std::array<double, 2> {
/** Initializes with the zero vector
 */
    FieldType();
/** Given a uniform value in [0,1), initializes a unit vector in a random direction
 */
    FieldType(double uniform);
/** Initializes vector with components a and b
 */
    FieldType(double a, double b);
/** Inner product
 */
    double operator*(FieldType b) const;
/** Scalar division
 */
    FieldType operator/(double b) const;
/** Scalar multiplication
 */
    FieldType operator*(int b) const;
/** Scalar multiplication
 */
    FieldType operator*(double b) const;
/** Vector addition
 */
    FieldType operator+(FieldType b) const;
/** Negation
 */
    FieldType operator-() const;
/** Vector subtraction
 */
    FieldType operator-(FieldType b) const;

    FieldType operator*=(double b);

    FieldType operator/=(double b);

    FieldType operator+=(FieldType b);

    FieldType operator-=(FieldType b);

};


using EdgeType = double;
using VertexType = FieldType;
const double kEpsilon = 1e-13;


FieldType operator*(const double& a, const FieldType& b);

FieldType operator*(const int& a, const FieldType& b);
}

#include "types.inl"


    
