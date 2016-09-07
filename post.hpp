#include "Eigen/Dense"

template<typename T> inline auto Bin(Eigen::Matrix<T, Eigen::Dynamic, 1> samples) -> 
    std::enable_if_t<std::is_arithmetic<T>::value, decltype(samples)> {
        return Eigen::Map<Eigen::Matrix<double, 2, Eigen::Dynamic>>(samples.data(), 2, samples.size()/2).transpose().rowwise().sum().array() / 2;
}


