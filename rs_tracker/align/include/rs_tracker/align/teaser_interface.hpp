#pragma once

#include <Eigen/Geometry>

namespace rs_tracker {

/**
 * @brief         Interface to TEASER registration.
 *
 * @param[in]     src_cloud Source point cloud.
 * @param[in]     dst_cloud Target point cloud.
 * @param[in,out] transform Transform; value will be used as initial guess.
 *
 * @return if the registration was successful.
 */
bool RegisterTeaser(const Eigen::Matrix<float, 3, Eigen::Dynamic>& src_cloud_vv,
                    const Eigen::Matrix<float, 3, Eigen::Dynamic>& dst_cloud_vv,
                    const Eigen::Matrix<float, 33, Eigen::Dynamic>& src_fpfh,
                    const Eigen::Matrix<float, 33, Eigen::Dynamic>& dst_fpfh,
                    const float noise_bound,
                    Eigen::Isometry3f* const transform);

}  // namespace rs_tracker
