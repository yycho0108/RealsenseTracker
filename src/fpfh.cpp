#include "rs_tracker/fpfh.hpp"

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/SVD>

#include <fmt/printf.h>
#include <omp.h>

#include "rs_tracker/point_cloud_utils.hpp"

namespace rs_tracker {

static constexpr const int kNumBins{11};
static constexpr const int kFpfhSize{3 * kNumBins};
static constexpr const int kSymmetricPfh{true};

using Vector33f = Eigen::Matrix<float, 33, 1>;
using KdTree33f = KDTreeChoCloudAdaptor<float, 33>;

static bool ComputePfh(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1,
                       const Eigen::Vector3f& p2, const Eigen::Vector3f& n2,
                       Eigen::Vector3f* const f) {
  // Compute normalized displacement between p1 and p2.
  Eigen::Vector3f delta = p2 - p1;
  const float distance = delta.norm();
  if (distance == 0.0f) {
    f->setZero();
    return false;
  }
  delta *= (1.0f / distance);

  // Make sure the same point is selected as 1 and 2 for each pair
  const float n1_d = n1.dot(delta);
  const float n2_d = n2.dot(delta);

  float u_d, nt_d;
  if (kSymmetricPfh) {
    // NOTE(yycho0108): The original paper switches the "origin"
    // based on the angle with the displacement vector. why?
    if (std::abs(n1_d) < std::abs(n2_d)) {
      // Switch p1 and p2.
      u_d = -n2_d;
      nt_d = -n1_d;
    } else {
      u_d = n1_d;
      nt_d = n2_d;
    }
  } else {
    u_d = n1_d;
    nt_d = n2_d;
  }

  if (std::abs(u_d) >= 1.0) {
    f->setZero();
    return false;
  }

  const float v_norm = std::sqrt(1 - u_d * u_d);
  const float n1n2 = n1.dot(n2);
  // f->coeffRef(0) = std::atan((nt_d - n1n2 * u_d)/(n1n2 * v_norm));  // -{}
  f->coeffRef(0) = std::atan2(nt_d - n1n2 * u_d, n1n2 * v_norm);  // f4
  f->coeffRef(1) = delta.dot(n1.cross(n2)) / v_norm;              // f1
  f->coeffRef(2) = u_d;                                           // f3

  return true;
}

static void ComputeSpfh(const Cloud3f& points, const Cloud3f& normals,
                        const std::vector<std::pair<int, float>>& nbrs,
                        const int index,
                        Eigen::Matrix<float, kFpfhSize, 1>* const spfh) {
  // Constant params
  // static const std::array<float, 3> scale{1.0 / M_PI, 0.5, 0.5};
  static const std::array<float, 3> scale{1.0 / (2 * M_PI), 0.5, 0.5};
  // NOTE(yycho0108): skip self (nbrs is expected to include `self`).
  const float dhist = 1.0f / (nbrs.size() - 1);

  Eigen::Vector3f pfh;
  spfh->setZero();
  for (const auto& entry : nbrs) {
    const int nbr_i = entry.first;
    // Skip self.
    if (nbr_i == index) {
      continue;
    }

    const bool suc =
        ComputePfh(points.GetPoint(index), normals.GetPoint(index),
                   points.GetPoint(nbr_i), normals.GetPoint(nbr_i), &pfh);
    if (suc) {
      for (int j = 0; j < 3; ++j) {
        const int h_index_raw = static_cast<int>(
            std::floor(kNumBins * ((pfh[j] * scale[j]) + 0.5)));
        const int h_index = std::max(0, std::min(kNumBins - 1, h_index_raw));
        spfh->coeffRef(j * kNumBins + h_index) += dhist;
      }
    }
  }

#if 0
  // NOTE(yycho0108): The following code block can be disabled since (in
  // general) feature computation will not fail.
  // TODO(yycho0108): the following code block can be made more efficient.
  for (int k = 0; k < 3; ++k) {
    const float quotient = spfh->segment(k * kNumBins, kNumBins).sum();
    if (quotient > 0) {
      spfh->segment(k * kNumBins, kNumBins) *= (1.0f / quotient);
    }
  }
#endif
}

void ComputeFpfhImpl(const Cloud3f& points, const Cloud3f& normals,
                     const KDTree3f& tree, const float radius,
                     const int num_neighbors, Cloud33f* const fpfh_out) {
  // NOTE(yycho0108): Actual num neighbors + self (included in tree)
  // TODO(yycho0108): Consider `exclude_self` options in search containers?
  // std::vector<int> nbrs(num_neighbors + 1);

  const int n = points.GetNumPoints();

  // Allocate output container.
  fpfh_out->SetNumPoints(n);

  const float radius_sq{radius * radius};

  // Keep track of intermediary SPFH values.
  std::vector<Eigen::Matrix<float, kFpfhSize, 1>> spfhs(n);

  std::vector<std::pair<int, float>> nbrs;
  // #pragma omp parallel for private(nbrs)
  for (std::size_t i = 0; i < n; ++i) {
    // FIXME(yycho0108): Revert to radius neighbors?
    // NOTE(yycho0108): I think using k-neighbors is actually fine for
    // obstacle-like point clouds
    const Vector3f& p = points.GetPoint(i);
    tree.index->radiusSearch(p.data(), radius_sq, nbrs, false);
    ComputeSpfh(points, normals, nbrs, i, &spfhs[i]);
  }

  // Combine spfh to compute final fpfh value.
  // #pragma omp parallel for private(nbrs)
  for (std::size_t i = 0; i < n; ++i) {
    // Recompute neighbors.
    // TODO(yycho0108): radius or knn?
    const Vector3f& p = points.GetPoint(i);
    tree.index->radiusSearch(p.data(), radius_sq, nbrs, false);

    // Compute features.
    auto feat = fpfh_out->GetPoint(i);

    // Appears to be of no consequence?
    feat.setZero();  // PCL version : skip self.
    // feat = spfhs[i];  // Paper version : include self.

    for (const auto& entry : nbrs) {
      const int nbr_i = entry.first;
      // skip self.
      if (nbr_i == static_cast<int>(i)) {
        continue;
      }

      const float dist = std::sqrt(entry.second);
      feat += (1.0F / dist) * spfhs[nbr_i];
    }

    // Normalize each histogram segment, to sum to 1.
    for (int k = 0; k < 3; ++k) {
      const float quotient = feat.segment(k * kNumBins, kNumBins).sum();
      if (quotient > 0) {
        feat.segment(k * kNumBins, kNumBins) *= (1.0f / quotient);
      }
    }
  }
}

void ComputeFpfhFasterImpl(const Cloud3f& points, const Cloud3f& normals,
                           const KDTree3f& tree, const float radius,
                           const int num_neighbors, Cloud33f* const fpfh_ptr) {
  static_cast<void>(radius);
  const int n = points.GetNumPoints();
  // NOTE(yycho0108): Actual num neighbors + self (included in tree)
  // TODO(yycho0108): Consider `exclude_self` options in search containers?
  std::vector<std::pair<int, float>> nbrs;

  // Allocate output container.
  auto& fpfhs = *fpfh_ptr;

  // TODO(yycho0108): Perhaps there's a better way to zero out the feature
  // vectors.
  fpfhs.SetNumPoints(n);
  fpfhs.GetData().setZero();

  for (std::size_t i = 0; i < n; ++i) {
    const Vector3f& p = points.GetPoint(i);
    // NOTE(yycho0108): I think using k-neighbors is actually fine for
    // obstacle-like point clouds?
    Vector33f spfh;
#if 1
    tree.index->radiusSearch(p.data(), radius * radius, nbrs, false);
#else
    tree.SearchKNeighbors(points[i].data(), num_neighbors, &nbrs, false);
#endif
    ComputeSpfh(points, normals, nbrs, i, &spfh);

#if 1
    // TODO(yycho0108): Figure out if this is necessary.
    fpfhs.GetPoint(i) += spfh;
#endif

    // Radius neighbor is a symmetric relation, thus distribute out to neighbors
    // here.
    for (const auto& entry : nbrs) {
      const int nbr_i = entry.first;
      if (nbr_i == static_cast<int>(i)) {
        continue;
      }
      // TODO(yycho0108): Consider how this weight term responds relative to the
      // (1.0) weight.
      const float weight =
          1.0 / (points.GetPoint(i) - points.GetPoint(nbr_i)).norm();
      fpfhs.GetPoint(nbr_i) += weight * spfh;
    }
  }

  // Normalize the final fpfh values.
  for (std::size_t i = 0; i < n; ++i) {
    auto feat = fpfhs.GetPoint(i);
    for (int k = 0; k < 3; ++k) {
      const float quotient = feat.segment(k * kNumBins, kNumBins).sum();
      if (quotient > 0) {
        feat.segment(k * kNumBins, kNumBins) *= (1.0f / quotient);
      }
    }
  }
}
void ComputeFpfh(const Cloud3f& cloud, const Vector3f& viewpoint,
                 const int normal_k, const float feature_radius,
                 Cloud33f* const fpfh_out) {
  // 1) Build indexable KdTree.
  const KDTree3f tree{std::cref(cloud), 16};
  // OINFO("TREE>{}", w_tree.Stop().Microseconds());

  // 2) Compute normals and also orient them.
  Cloud3f normals;
  ComputeNormals(cloud, tree, normal_k, &normals);
  OrientNormals(cloud, viewpoint, &normals);

  // 3) After precomputing relevant quantities, compute fpfh features.
  ComputeFpfhImpl(cloud, normals, tree, feature_radius, 0, fpfh_out);
  // ComputeFpfhFasterImpl(cloud, normals, tree, radius, num_neighbors,
  // fpfh_out); OINFO("FPFH>{}", w_fpfh.Stop().Microseconds());
}

Cloud33f ComputeFpfh(const Cloud3f& cloud, const Vector3f& viewpoint,
                     const int normal_k, const float feature_radius) {
  Cloud33f out;
  ComputeFpfh(cloud, viewpoint, normal_k, feature_radius, &out);
  return out;
}

void ComputeMatch(const Cloud33f& src, const Cloud33f& dst,
                  Eigen::VectorXi* const matches) {
  const KdTree33f tree{std::cref(dst), 16};
  matches->resize(src.GetNumPoints());
  for (std::size_t i = 0; i < src.GetNumPoints(); ++i) {
    const Vector33f& p = src.GetPoint(i);
    int oi;
    float odsq;
    tree.index->knnSearch(p.data(), 1, &oi, &odsq);
    matches->coeffRef(i) = oi;
  }
}

Eigen::VectorXi ComputeMatch(const Cloud33f& src, const Cloud33f& dst) {
  Eigen::VectorXi matches;
  ComputeMatch(src, dst, &matches);
  return matches;
}

void ComputeMatches(const Cloud33f& src, const Cloud33f& dst,
                    const int num_matches, Eigen::MatrixXi* const matches) {
  const KdTree33f tree{std::cref(dst), 16};
  std::vector<int> knn_indices(num_matches);
  std::vector<float> knn_squared_distances(num_matches);
  matches->resize(src.GetNumPoints(), num_matches);
  for (std::size_t i = 0; i < src.GetNumPoints(); ++i) {
    const Vector33f& p = src.GetPoint(i);
    tree.index->knnSearch(p.data(), num_matches, knn_indices.data(),
                          knn_squared_distances.data());
    for (int j = 0; j < num_matches; ++j) {
      matches->coeffRef(i, j) = knn_indices[j];
    }
  }
}

Eigen::MatrixXi ComputeMatches(const Cloud33f& src, const Cloud33f& dst,
                               const int num_matches) {
  Eigen::MatrixXi out;
  ComputeMatches(src, dst, num_matches, &out);
  return out;
}

}  // namespace rs_tracker
