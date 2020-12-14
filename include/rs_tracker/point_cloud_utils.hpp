#pragma once

#include <cho_util/core/geometry/point_cloud.hpp>

#include "rs_tracker/common.hpp"

namespace rs_tracker {

void DownsampleVoxel(const cho::core::PointCloud<float, 3>& cloud_in,
                     const float voxel_size,
                     cho::core::PointCloud<float, 3>* const cloud_out);

void FindCorrespondences(const KDTree& tree,
                         const cho::core::PointCloud<float, 3>& source,
                         std::vector<int>* const indices,
                         std::vector<float>* const squared_distances);

void ComputeCentroid(const cho::core::PointCloud<float, 3>& cloud,
                     Eigen::Vector3f* const centroid);

void ComputeCovariances(const KDTree& tree,
                        const cho::core::PointCloud<float, 3>& cloud,
                        std::vector<Eigen::Matrix3f>* const covs,
                        const bool use_gicp);

void RemoveNans(const cho::core::PointCloud<float, 3>& cloud_in,
                cho::core::PointCloud<float, 3>* const cloud_out);

}  // namespace rs_tracker
