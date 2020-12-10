#pragma once

#include <vector>

#include <Eigen/Core>

#include <cho_util/core/geometry//point_cloud.hpp>

#include "rs_tracker/common.hpp"
#include "rs_tracker/gicp_cost.hpp"

namespace rs_tracker {

// ceres::Solver::Options GetOptions();
float ComputeAlignment(const cho::core::PointCloud<float, 3>& src,
                       const cho::core::PointCloud<float, 3>& dst,
                       const std::vector<Eigen::Matrix3f>& src_covs,
                       const std::vector<Eigen::Matrix3f>& dst_covs,
                       const std::vector<int>& dst_indices,
                       const Eigen::Isometry3f& seed,
                       Eigen::Isometry3f* const transform);

float ComputeAlignment(const cho::core::PointCloud<float, 3>& src,
                       const cho::core::PointCloud<float, 3>& dst,
                       Eigen::Isometry3f* const transform);

}  // namespace rs_tracker
