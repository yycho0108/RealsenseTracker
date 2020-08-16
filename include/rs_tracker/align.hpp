#pragma once

#include <Eigen/src/Geometry/Transform.h>
#include <vector>

#include "rs_tracker/common.hpp"
#include "rs_tracker/gicp_cost.hpp"

namespace rs_tracker {

// ceres::Solver::Options GetOptions();
float ComputeAlignment(const Cloud& src, const Cloud& dst,
                       const std::vector<Eigen::Matrix3f>& src_covs,
                       const std::vector<Eigen::Matrix3f>& dst_covs,
                       const std::vector<int>& dst_indices,
                       const Eigen::Isometry3f& seed,
                       Eigen::Isometry3f* const transform);

float ComputeAlignment(const Cloud& src, const Cloud& dst,
                       Eigen::Isometry3f* const transform);

}  // namespace rs_tracker
