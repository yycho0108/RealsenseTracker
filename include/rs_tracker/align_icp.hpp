#pragma once

#include <optional>
#include <vector>

#include <Eigen/Geometry>

#include <cho_util/core/geometry/point_cloud.hpp>

#include "rs_tracker/common.hpp"

namespace rs_tracker {

bool SolveKabsch(const Cloud3f& src, const Cloud3f& dst,
                 const std::vector<std::pair<int, int>>& indices,
                 const std::vector<float>& weights,
                 Eigen::Isometry3f* const xfm);

bool AlignIcp3d(const Cloud3f& src, const Cloud3f& dst,
                const KDTree3f& dst_tree, const int max_iter,
                Eigen::Isometry3f* const transform);

bool AlignIcp3d(const Cloud3f& src, const Cloud3f& dst, const int max_iter,
                Eigen::Isometry3f* const transform);

}  // namespace rs_tracker
