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

bool SolveKabsch(const Cloud3f& src, const Cloud3f& dst,
                 const std::vector<std::pair<int, int>>& indices,
                 const std::vector<float>& weights,
                 Eigen::Isometry3f* const xfm) {
  // Skip insufficient number of points ...
  if (src.GetNumPoints() < 3 || dst.GetNumPoints() < 3) {
    return false;
  }

  // Compute centroids.
  Eigen::Vector3f src_mean{Eigen::Vector3f::Zero()};
  Eigen::Vector3f dst_mean{Eigen::Vector3f::Zero()};
  for (const auto& im : indices) {
    src_mean += src.GetPoint(im.first);
    dst_mean += dst.GetPoint(im.second);
  }
  src_mean /= indices.size();
  dst_mean /= indices.size();

  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  if (weights.empty()) {
    for (const auto& im : indices) {
      const auto& p_src = src.GetPoint(im.first);
      const auto& p_dst = dst.GetPoint(im.second);

      cov +=
          ((p_dst - dst_mean) * (p_src - src_mean).transpose()).cast<double>();
    }
  } else {
    int c{0};
    for (const auto& im : indices) {
      const auto& p_src = src.GetPoint(im.first);
      const auto& p_dst = dst.GetPoint(im.second);
      cov +=
          weights[c] *
          ((p_dst - dst_mean) * (p_src - src_mean).transpose()).cast<double>();
      ++c;
    }
  }
  // Solve Rotation.
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd{
      cov, Eigen::ComputeFullU | Eigen::ComputeFullV};
  Eigen::Matrix3f R = (svd.matrixU() * svd.matrixV().transpose()).cast<float>();
  if (R.determinant() < 0) {
    R.col(2) *= -1;
  }

  // Solve Translation - assuming rotation about origin {0,0,0}.
  const Eigen::Vector3f translation = dst_mean - (R * src_mean);

  // Compose transform.
  *xfm = Eigen::Translation3f{translation} * Eigen::Quaternionf{R};
  return true;
}

bool AlignIcp3d(const Cloud3f& src, const Cloud3f& dst,
                const KDTree3f& dst_tree, const int max_iter,
                Eigen::Isometry3f* const transform) {
  // Skip insufficient number of points ...
  if (src.GetNumPoints() < 3 || dst.GetNumPoints() < 3) {
    return false;
  }

  cho::util::UTimer timer_src_mean{true};
  Eigen::Isometry3f xfm = *transform;
  const int n = src.GetNumPoints();
  fmt::print("src size : {}\n", n);
  const Eigen::Vector3f src_mean =
      ReturnLastPtr<Eigen::Vector3f>(ComputeCentroid, src);
  fmt::print("src_mean : {} us\n", timer_src_mean.StopAndGetElapsedTime());
  fmt::print("max iter = {}\n", max_iter);

  float cost{0};
  float mu{1.0f};
  for (int iter = 0; iter < max_iter; ++iter) {
    cho::util::UTimer timer_iter{true};

    // GNC-type
    if (iter > 0 && iter % 8 == 0) {
      mu /= 1.4f;
    }

    // Compute correspondences.
    Eigen::Vector3f dst_mean = Eigen::Vector3f::Zero();
    std::vector<int> nbrs(n);
    std::vector<float> weights(n);
    cost = 0;
    for (int i = 0; i < n; ++i) {
      // Apply transform.
      const Eigen::Vector3f p = xfm * src.GetPoint(i);

      // Compute correspondence.
      float dist_sqr{0};
      int j{0};
      dst_tree.query(p.data(), 1, &j, &dist_sqr);
      cost += dist_sqr;
      nbrs[i] = j;

      const float l_pq_rt = mu / (dist_sqr + mu);
      const float l_pq = l_pq_rt * l_pq_rt;
      weights[i] = l_pq;

      dst_mean += dst.GetPoint(j);
    }
    dst_mean /= n;

    // Accumulate covariance.
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int i = 0; i < n; ++i) {
      if (1) {
        cov += (weights[i] * (dst.GetPoint(nbrs[i]) - dst_mean) *
                (src.GetPoint(i) - src_mean).transpose())
                   .cast<double>();
      } else {
        cov += ((dst.GetPoint(nbrs[i]) - dst_mean) *
                (src.GetPoint(i) - src_mean).transpose())
                   .cast<double>();
      }
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
  fmt::print("mean cost = {}\n", mean_cost);
  // return mean_cost < 0.085f;
  return mean_cost < 10000;
}

bool AlignIcp3d(const Cloud3f& src, const Cloud3f& dst, const int max_iter,
                Eigen::Isometry3f* const transform) {
  const KDTree3f dst_tree{std::cref(dst), 16};
  return AlignIcp3d(src, dst, dst_tree, max_iter, transform);
}

}  // namespace rs_tracker
