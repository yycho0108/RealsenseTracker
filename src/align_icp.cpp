#include "rs_tracker/align_icp.hpp"

#include <fmt/printf.h>

#include <cho_util/util/timer.hpp>

#include "rs_tracker/point_cloud_utils.hpp"

namespace rs_tracker {

template <typename Out, typename... Args, typename Function>
Out ReturnLastPtr(Function f, Args... args) {
  Out out;
  f(args..., &out);
  return out;
}

bool AlignIcp3d(const cho::core::PointCloud<float, 3>& src,
                const cho::core::PointCloud<float, 3>& dst,
                const KDTree& dst_tree, Eigen::Isometry3f* const transform) {
  // Skip insufficient number of points ...
  if (src.GetNumPoints() < 3 || dst.GetNumPoints() < 3) {
    return false;
  }

  cho::util::UTimer timer_src_mean{true};
  Eigen::Isometry3f xfm = *transform;
  fmt::print("src size : {}\n", src.GetNumPoints());
  const Eigen::Vector3f src_mean =
      ReturnLastPtr<Eigen::Vector3f>(ComputeCentroid, src);
  fmt::print("src_mean : {} us\n", timer_src_mean.StopAndGetElapsedTime());

  float cost{0};
  float mu{2.0f};
  for (int iter = 0; iter < 128; ++iter) {
    cho::util::UTimer timer_iter{true};

    // GNC
    if (iter > 0 && iter % 4 == 0) {
      mu /= 1.4f;
    }

    // Compute correspondences.
    Eigen::Vector3f dst_mean = Eigen::Vector3f::Zero();
    std::vector<int> nbrs(src.GetNumPoints());
    std::vector<float> weights(src.GetNumPoints());
    cost = 0;
    for (int i = 0; i < src.GetNumPoints(); ++i) {
      // Apply transform.
      const Eigen::Vector3f p = xfm * src.GetPoint(i);

      // Compute correspondence.
      float dist_sqr{0};
      int j{0};
      dst_tree.query(p.data(), 1, &j, &dist_sqr);
      cost += dist_sqr;
      nbrs[i] = j;
      // weights[i] = std::exp(-dist_sqr) / std::exp(-0.1);

      // const float l_pq_rt = mu / (dist_sqr + mu);
      // const float l_pq = l_pq_rt * l_pq_rt;
      // weights[i] = l_pq;
      weights[i] = std::exp(-dist_sqr) / std::exp(-0.1);

      dst_mean += dst.GetPoint(j);
    }
    dst_mean /= src.GetNumPoints();

    // Accumulate covariance.
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int i = 0; i < src.GetNumPoints(); ++i) {
      cov += (weights[i] * (dst.GetPoint(nbrs[i]) - dst_mean) *
              (src.GetPoint(i) - src_mean).transpose())
                 .cast<double>();
    }

    // Solve Rotation.
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd{
        cov, Eigen::ComputeFullU | Eigen::ComputeFullV};
    Eigen::Matrix3f R =
        (svd.matrixU() * svd.matrixV().transpose()).cast<float>();
    if (R.determinant() < 0) {
      R.col(2) *= -1;
    }

    // Solve Translation - assuming rotation about origin {0,0,0}.
    const Eigen::Vector3f translation = dst_mean - (R * src_mean);

    // Compose transform.
    xfm = Eigen::Translation3f{translation} * Eigen::Quaternionf{R};
    // fmt::print("iter : {} us\n", timer_iter.StopAndGetElapsedTime());
  }

  // Output transform.
  *transform = xfm;
  const float mean_cost = std::sqrt(cost / src.GetNumPoints());
  fmt::print("mean cost = {}", mean_cost);
  // return mean_cost < 0.085f;
  return mean_cost < 10000;
}

bool AlignIcp3d(const cho::core::PointCloud<float, 3>& src,
                const cho::core::PointCloud<float, 3>& dst,
                Eigen::Isometry3f* const transform) {
  const KDTree dst_tree{3, std::cref(dst), 16};
  return AlignIcp3d(src, dst, dst_tree, transform);
}

}  // namespace rs_tracker
