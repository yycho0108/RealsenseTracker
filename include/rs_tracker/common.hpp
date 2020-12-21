#pragma once

#include <Eigen/Core>

#include <nanoflann.hpp>

#include <cho_util/core/geometry/point_cloud.hpp>

#include "rs_tracker/kdtree.hpp"

namespace rs_tracker {

using KDTree3f = KDTreeChoCloudAdaptor<float, 3>;

template <typename T>
using Vector3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using MapVector3 = Eigen::Map<Vector3<T>>;

using Vector3f = Eigen::Vector3f;

using Cloud3f = cho::core::PointCloud<float, 3>;

}  // namespace rs_tracker
