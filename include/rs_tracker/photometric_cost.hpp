#pragma once

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <functional>

#include <ceres/autodiff_cost_function.h>
#include <ceres/autodiff_local_parameterization.h>
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <ceres/loss_function.h>
#include <ceres/rotation.h>
#include <ceres/types.h>

namespace rs_tracker {

struct PhotometricCost {
  using RefVec3f = Eigen::Ref<const Eigen::Vector3f>;
  using RefMat3f = Eigen::Ref<const Eigen::Matrix3f>;

 private:
  Eigen::Vector3f src_pt_;
  float src_col_;
  std::reference_wrapper<const Eigen::MatrixXf> dst_img_;

 public:
  explicit PhotometricCost(
      const Eigen::Vector3f& src_pt, const float src_col,
      const std::reference_wrapper<Eigen::MatrixXf>& dst_img)
      : src_pt_(src_pt), src_col_(src_col), dst_img_(dst_img) {}
  /**
   * Factory to hide the construction of the CostFunction object from the client
   * code.
   */
  template <typename... Args>
  static ceres::CostFunction* Create(Args&&... args) {
    static constexpr const int kNumResidual = 1;
    static constexpr const int kNumRotParam = 4;
    static constexpr const int kNumTransParam = 3;
    return (new ceres::AutoDiffCostFunction<PhotometricCost, kNumResidual,
                                            kNumRotParam, kNumTransParam>(
        new PhotometricCost(args...)));
  }

  template <typename T>
  bool operator()(const T* const p_rxn, const T* const p_txn,
                  T* const p_err) const {
    static constexpr const int nbr_size = 3;

    // Unpack parameters.
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> txn{p_txn};
    const Eigen::Map<const Eigen::Quaternion<T>> qxn{p_rxn};

    // Project to target frame and obtain corresponding color,
    // retrieved through "local color function".
    const Eigen::Matrix<T, 3, 1> dst_pt{qxn * src_pt_ + txn};
    T color_dst = GetColor(dst_img_, project(qxn * src_pt_ + txn), nbr_size);

    // Output residual.
    *p_err = color_dst - src_col_;
    return true;
  }
};

}  // namespace rs_tracker
