#pragma once

#include <cho_util/core/geometry/point_cloud.hpp>

#include "rs_tracker/common/types.hpp"

namespace rs_tracker {

void DownsampleVoxel(const Cloud3f& cloud_in, const float voxel_size,
                     Cloud3f* const cloud_out);

void FindCorrespondences(const KDTree3f& tree, const Cloud3f& source,
                         std::vector<int>* const indices,
                         std::vector<float>* const squared_distances);

void ComputeCentroid(const Cloud3f& cloud, Eigen::Vector3f* const centroid);

void ComputeCovariances(const KDTree3f& tree, const Cloud3f& cloud,
                        std::vector<Eigen::Matrix3f>* const covs,
                        const bool use_gicp);

void RemoveNans(const Cloud3f& cloud_in, Cloud3f* const cloud_out);

void ComputeNormals(const Cloud3f& cloud, const KDTree3f& tree,
                    const float num_neighbors, Cloud3f* const normals);

void OrientNormals(const Cloud3f& cloud, const Vector3f& viewpoint,
                   Cloud3f* const normals);

}  // namespace rs_tracker
