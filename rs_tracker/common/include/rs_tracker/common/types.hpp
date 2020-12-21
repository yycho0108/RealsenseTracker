#pragma once

#include <Eigen/Core>

#include <cho_util/core/geometry/point_cloud.hpp>

#include "rs_tracker/common/kdtree.hpp"

namespace rs_tracker {

using Vector3f = Eigen::Vector3f;
using Matrix3f = Eigen::Matrix3f;

using Cloud3f = cho::core::PointCloud<float, 3>;
using KDTree3f = KDTreeChoCloudAdaptor<float, 3>;

using Cloud33f = cho::core::PointCloud<float, 33>;
using KDTree33f = KDTreeChoCloudAdaptor<float, 33>;

using Matrix33Xf = Eigen::Matrix<float, 33, Eigen::Dynamic>;

}  // namespace rs_tracker
