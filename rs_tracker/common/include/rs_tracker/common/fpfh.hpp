#pragma once

#include <vector>

#include <cho_util/core/geometry.hpp>

#include "rs_tracker/common/types.hpp"

namespace rs_tracker {

void ComputeFpfh(const cho::core::PointCloud<float, 3>& cloud,
                 const Eigen::Vector3f& viewpoint, const int normal_k,
                 const float feature_radius,
                 cho::core::PointCloud<float, 33>* const fpfh_out);

Cloud33f ComputeFpfh(const cho::core::PointCloud<float, 3>& cloud,
                     const Eigen::Vector3f& viewpoint, const float normal_k,
                     const float feature_radius);

void ComputeMatch(const Cloud33f& src, const Cloud33f& dst,
                  Eigen::VectorXi* const matches);

Eigen::VectorXi ComputeMatch(const Cloud33f& src, const Cloud33f& dst);

void ComputeMatches(const Cloud33f& src, const Cloud33f& dst,
                    const int num_matches, Eigen::MatrixXi* const matches);

Eigen::MatrixXi ComputeMatches(const Cloud33f& src, const Cloud33f& dst,
                               const int num_matches);

}  // namespace rs_tracker
