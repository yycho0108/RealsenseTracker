#include "rs_tracker/common/point_cloud_utils.hpp"

#include <tuple>
#include <unordered_map>

#include <fmt/printf.h>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <boost/functional/hash.hpp>

template <typename T>
struct MatrixHash : std::unary_function<T, std::size_t> {
  std::size_t operator()(T const& mat) const noexcept {
    std::size_t seed = 0;
    for (int i = 0; i < mat.size(); ++i) {
      boost::hash_combine(seed, std::hash<typename T::Scalar>()(mat(i)));
    }
    return seed;
  }
};

namespace rs_tracker {

void ComputeExtents(const Cloud3f& cloud, Eigen::AlignedBox3f* const box) {
  box->setEmpty();
  for (int i = 0; i < cloud.GetNumPoints(); ++i) {
    const auto& p = cloud.GetPoint(i);
    box->extend(p);
  }
}

void DownsampleVoxel(const Cloud3f& cloud_in, const float voxel_size,
                     Cloud3f* const cloud_out) {
  std::unordered_map<Eigen::Vector3i, int, MatrixHash<Eigen::Vector3i>> vox;

  // Vox
  for (int i = 0; i < cloud_in.GetNumPoints(); ++i) {
    const Eigen::Vector3f& point = cloud_in.GetPoint(i);
    const Eigen::Vector3i index =
        (point.array() / voxel_size).floor().template cast<int>();
    auto it = vox.find(index);
    if (it == vox.end()) {
      vox.emplace(index, i);
    }
  }

  // Shallow aliasing check.
  if (cloud_out == &cloud_in) {
    Cloud3f tmp;
    tmp.SetNumPoints(vox.size());
    int i = 0;
    for (const auto& p : vox) {
      tmp.GetPoint(i) = cloud_in.GetPoint(p.second);
      ++i;
    }
    *cloud_out = std::move(tmp);
  } else {
    // Out
    cloud_out->SetNumPoints(vox.size());
    int i = 0;
    for (const auto& p : vox) {
      cloud_out->GetPoint(i) = cloud_in.GetPoint(p.second);
      ++i;
    }
  }
}

void FindCorrespondences(const KDTree3f& tree, const Cloud3f& source,
                         std::vector<int>* const indices,
                         std::vector<float>* const squared_distances) {
  const int n = source.GetNumPoints();
  // Reset memory.
  indices->clear();
  squared_distances->clear();
  indices->reserve(n);
  squared_distances->reserve(n);

  // Temps
  int knn_indices[1];
  float knn_squared_distances[1];
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3f& query_point = source.GetPoint(i);
    tree.index->knnSearch(reinterpret_cast<const float*>(query_point.data()), 1,
                          knn_indices, knn_squared_distances);
    indices->emplace_back(knn_indices[0]);
    squared_distances->emplace_back(knn_squared_distances[0]);
  }
}

void ComputeCentroid(const Cloud3f& cloud, Eigen::Vector3f* const centroid) {
  centroid->setZero();
  for (int i = 0; i < cloud.GetNumPoints(); ++i) {
    *centroid += cloud.GetPoint(i);
  }
  *centroid *= (1.0 / cloud.GetNumPoints());
}

void ComputeCovariances(const KDTree3f& tree, const Cloud3f& cloud,
                        std::vector<Eigen::Matrix3f>* const covs,
                        const bool use_gicp) {
  // Set default number of neighbors and allocate containers.
  constexpr const int kNeighbors = 32;
  std::vector<int> knn_indices(kNeighbors + 1);
  std::vector<float> knn_squared_distances(kNeighbors + 1);

  const int n = cloud.GetNumPoints();
  // Allocate covariance output.
  covs->resize(n);
  for (int i = 0; i < n; ++i) {
    // Extract input and output pair.
    const Eigen::Vector3f& query_point = cloud.GetPoint(i);
    Eigen::Matrix3f& cov = covs->at(i);

    // Search neighbors.
    tree.index->knnSearch(reinterpret_cast<const float*>(query_point.data()),
                          kNeighbors + 1, knn_indices.data(),
                          knn_squared_distances.data());

    // Compute centroid.
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (int j = 1; j < kNeighbors + 1; ++j) {
      const int index = knn_indices[j];
      centroid += cloud.GetPoint(index);
    }
    centroid /= (kNeighbors);

    // Compute covariance.
    cov.setZero();
    for (int j = 1; j < kNeighbors + 1; ++j) {
      const int index = knn_indices[j];
      const Eigen::Vector3f delta = cloud.GetPoint(index) - centroid;
      cov += delta * delta.transpose();
    }

    // Remap covariance to GICP form.
    // In this case, division is not necessary.
    if (use_gicp) {
      // Compute the SVD (covariance matrix is symmetric so U = V')
      // TODO(yycho0108): Maybe a more efficient form exists ?
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU);
      Eigen::Matrix3f U = svd.matrixU();
      cov.setZero();
      // Reconstitute the covariance matrix with modified singular values using
      // the column vectors in V.
      for (int k = 0; k < 3; k++) {
        Eigen::Vector3f col = U.col(k);
        double v = 1.;  // biggest 2 singular values replaced by 1
        if (k == 2) {
          // smallest singular value replaced by gicp_epsilon
          v = 1e-2;
        }
        cov += v * col * col.transpose();
      }
    } else {
      // Compute standard covariance.
      cov /= (kNeighbors - 1);
    }
  }
}

void RemoveNans(const Cloud3f& cloud_in, Cloud3f* const cloud_out) {
  cloud_out->SetNumPoints(cloud_in.GetNumPoints());
  int out_size{0};
  for (int i = 0; i < cloud_in.GetNumPoints(); ++i) {
    if (!cloud_in.GetPoint(i).allFinite()) {
      continue;
    }
    cloud_out->GetPoint(out_size) = cloud_in.GetPoint(i);
    ++out_size;
  }
  cloud_out->GetData().conservativeResize(3, out_size);
}

void ComputeNormals(const Cloud3f& cloud, const KDTree3f& tree,
                    const float num_neighbors, Cloud3f* const normals) {
  std::vector<int> oi(num_neighbors);
  std::vector<float> od_sqr(num_neighbors);

  normals->SetNumPoints(cloud.GetNumPoints());
  for (int i = 0; i < cloud.GetNumPoints(); ++i) {
    const Eigen::Vector3f& p = cloud.GetPoint(i);
    tree.index->knnSearch(p.data(), num_neighbors, oi.data(), od_sqr.data());

    // Centroid
    Vector3f centroid = Vector3f::Zero();
    for (const auto j : oi) {
      centroid += cloud.GetPoint(j);
    }
    centroid /= num_neighbors;

    // Covariance
    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (const auto j : oi) {
      const Vector3f delta = cloud.GetPoint(j) - centroid;
      cov.noalias() += delta * delta.transpose();
    }

    // Normal
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig{cov};
    normals->GetPoint(i) = eig.eigenvectors().col(0);
  }
}

void OrientNormals(const Cloud3f& cloud, const Vector3f& viewpoint,
                   Cloud3f* const normals) {
  const int n = cloud.GetNumPoints();
  for (int i = 0; i < n; ++i) {
    const Vector3f ray = cloud.GetPoint(i) - viewpoint;
    auto normal = normals->GetPoint(i);
    if (ray.dot(normal) > 0) {
      normal *= -1;
    }
  }
}

}  // namespace rs_tracker
