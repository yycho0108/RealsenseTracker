#pragma once

#include <vector>

#include <Eigen/Geometry>

#include <cho_util/core/geometry/point_cloud.hpp>

#include "rs_tracker/common.hpp"

namespace rs_tracker {

bool AlignIcp3d(const cho::core::PointCloud<float, 3>& src,
                const cho::core::PointCloud<float, 3>& dst,
                const KDTree& dst_tree, Eigen::Isometry3f* const transform);

bool AlignIcp3d(const cho::core::PointCloud<float, 3>& src,
                const cho::core::PointCloud<float, 3>& dst,
                Eigen::Isometry3f* const transform);

}  // namespace rs_tracker
