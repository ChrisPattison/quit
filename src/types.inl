#include <numeric>
#include <cmath>
#include <boost/math/constants/constants.hpp>

namespace propane {

inline FieldType::FieldType() {
    (*this)[0] = 0;
    (*this)[1] = 0;
}

inline FieldType::FieldType(ValueType uniform) {
    ValueType angle = uniform * boost::math::constants::pi<ValueType>();
    (*this)[0] = std::cos(angle);
    (*this)[1] = std::sin(angle);
}

inline FieldType::FieldType(ValueType a, ValueType b) {
    (*this)[0] = a;
    (*this)[1] = b;
}

__attribute__((always_inline))
inline auto FieldType::operator*(FieldType b) const -> ValueType {
    return (*this)[0] * b[0] + (*this)[1] * b[1];
}

inline FieldType FieldType::operator/(ValueType b) const {
        return (*this) * static_cast<ValueType>(1./b);
}

inline FieldType FieldType::operator*(int b) const {
    return {(*this)[0] * b, (*this)[1] * b};
}

inline FieldType FieldType::operator*(ValueType b) const {
    return {(*this)[0] * b, (*this)[1] * b};
}

inline FieldType FieldType::operator+(FieldType b) const {
    return {(*this)[0] + b[0], (*this)[1] + b[1]};
}

inline FieldType FieldType::operator-() const {
    return {-(*this)[0], -(*this)[1]};
}

inline FieldType FieldType::operator-(FieldType b) const {
    return {(*this)[0] - b[0], (*this)[1] - b[1]};
}

inline FieldType FieldType::operator*=(ValueType b) {
    (*this) = (*this) * b;
    return *this;
}

inline FieldType FieldType::operator/=(ValueType b) {
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

inline FieldType operator*(const float& a, const FieldType& b) {
    return b * a;
}
}
