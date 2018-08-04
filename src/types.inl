#include <numeric>
#include <cmath>
#include <boost/math/constants/constants.hpp>

namespace propane {

inline FieldType::FieldType() {
    (*this)[0] = 0;
    (*this)[1] = 0;
    (*this)[2] = 0;
}

inline FieldType::FieldType(double uniform1, double uniform2) {
    double angle1 = uniform1 * boost::math::constants::pi<double>();
    double angle2 = uniform2 * boost::math::constants::pi<double>();
    (*this)[0] = std::sin(angle1)*std::cos(angle2);
    (*this)[1] = std::sin(angle1)*std::sin(angle2);
    (*this)[2] = std::cos(angle1);
}

inline FieldType::FieldType(double a, double b, double c) {
    (*this)[0] = a;
    (*this)[1] = b;
    (*this)[2] = c;
}

inline double FieldType::operator*(FieldType b) const {
    return (*this)[0] * b[0] + (*this)[1] * b[1] + (*this)[2] * b[2];
}

inline FieldType FieldType::operator/(double b) const {
        return (*this) * (1./b);
}

inline FieldType FieldType::operator*(int b) const {
    return {(*this)[0] * b, (*this)[1] * b, (*this)[2] * b};
}

inline FieldType FieldType::operator*(double b) const {
    return {(*this)[0] * b, (*this)[1] * b, (*this)[2] * b};
}

inline FieldType FieldType::operator+(FieldType b) const {
    return {(*this)[0] + b[0], (*this)[1] + b[1], (*this)[2] + b[2]};
}

inline FieldType FieldType::operator-() const {
    return {-(*this)[0], -(*this)[1], -(*this)[2]};
}

inline FieldType FieldType::operator-(FieldType b) const {
    return {(*this)[0] - b[0], (*this)[1] - b[1], (*this)[2] - b[2]};
}

inline FieldType FieldType::operator*=(double b) {
    (*this) = (*this) * b;
    return *this;
}

inline FieldType FieldType::operator/=(double b) {
    (*this) = (*this) / b;
    return *this;
}

inline FieldType FieldType::operator+=(FieldType b) {
    (*this) = (*this) + b;
    return *this;
}

inline FieldType FieldType::operator-=(FieldType b) {
    (*this) = (*this) - b;
    return *this;
}


inline FieldType operator*(const int& a, const FieldType& b) {
    return b * a;
}

inline FieldType operator*(const double& a, const FieldType& b) {
    return b * a;
}
}
