#pragma once

#include <Eigen/Core>

#include <nanoflann.hpp>

namespace rs_tracker {

using Cloud = Eigen::MatrixX3f;

using KDTree = nanoflann::KDTreeEigenMatrixAdaptor<Cloud, 3>;

template <typename T>
using Vector3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using MapVector3 = Eigen::Map<Vector3<T>>;

using Vector3f = Eigen::Vector3f;

}  // namespace rs_tracker
