#pragma once

#include <Eigen/Core>

#include <nanoflann.hpp>

#include <cho_util/core/geometry/point_cloud.hpp>

namespace rs_tracker {

template <typename Scalar, int Dims, class Distance = nanoflann::metric_L2>
struct KDTreeChoCloudAdaptor {
  using cloud_t = cho::core::PointCloud<Scalar, Dims>;

  using self_t = KDTreeChoCloudAdaptor<Scalar, Dims, Distance>;
  using num_t = Scalar;
  using IndexType = int;
  using metric_t =
      typename Distance::template traits<num_t, self_t>::distance_t;
  using index_t =
      nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, Dims, IndexType>;

  index_t* index;  //! The kd-tree index for the user to call its methods as
                   //! usual with any other FLANN index.

  /// Constructor: takes a const ref to the matrix object with the data points
  explicit KDTreeChoCloudAdaptor(
      const size_t dimensionality,
      const std::reference_wrapper<const cloud_t>& cloud,
      const int leaf_max_size = 10)
      : m_cloud(cloud) {
#if 0
    // const auto dims = cloud.get().;
    if (size_t(dims) != dimensionality)
      throw std::runtime_error(
          "Error: 'dimensionality' must match column count in data matrix");
    if (Dims > 0 && int(dims) != Dims)
      throw std::runtime_error(
          "Data set dimensionality does not match the 'DIM' template argument");
#endif
    index =
        new index_t(Dims, *this /* adaptor */,
                    nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
    index->buildIndex();
  }

 public:
  /** Deleted copy constructor */
  KDTreeChoCloudAdaptor(const self_t&) = delete;

  ~KDTreeChoCloudAdaptor() { delete index; }

  const std::reference_wrapper<const cloud_t> m_cloud;

  /** Query for the \a num_closest closest points to a given point (entered as
   * query_point[0:dim-1]). Note that this is a short-cut method for
   * index->findNeighbors(). The user can also call index->... methods as
   * desired. \note nChecks_IGNORED is ignored but kept for compatibility with
   * the original FLANN interface.
   */
  inline void query(const num_t* query_point, const size_t num_closest,
                    IndexType* out_indices, num_t* out_distances_sq,
                    const int /* nChecks_IGNORED */ = 10) const {
    nanoflann::KNNResultSet<num_t, IndexType> resultSet(num_closest);
    resultSet.init(out_indices, out_distances_sq);
    index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
  }

  /** @name Interface expected by KDTreeSingleIndexAdaptor
   * @{ */

  const self_t& derived() const { return *this; }
  self_t& derived() { return *this; }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const {
    return m_cloud.get().GetNumPoints();
  }

  // Returns the dim'th component of the idx'th point in the class:
  inline num_t kdtree_get_pt(const IndexType idx, size_t dim) const {
    return m_cloud.get().GetPoint(idx)[dim];
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in
  //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
  //   the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const {
    return false;
  }

  /** @} */

};  // end of KDTreeChoCloudAdaptor

using KDTree = KDTreeChoCloudAdaptor<float, 3>;

template <typename T>
using Vector3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using MapVector3 = Eigen::Map<Vector3<T>>;

using Vector3f = Eigen::Vector3f;

}  // namespace rs_tracker
