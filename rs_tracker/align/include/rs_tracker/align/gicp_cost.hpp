#pragma once

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <ceres/autodiff_cost_function.h>
#include <ceres/autodiff_local_parameterization.h>
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <ceres/loss_function.h>
#include <ceres/rotation.h>
#include <ceres/types.h>

namespace rs_tracker {

struct GICPCost {
  using RefVec3f = Eigen::Ref<const Eigen::Vector3f>;
  using RefMat3f = Eigen::Ref<const Eigen::Matrix3f>;

  explicit GICPCost(const Eigen::Vector3f& src, const Eigen::Vector3f& dst,
                    const RefMat3f& src_cov, const RefMat3f& dst_cov)
      : src(src), dst(dst), src_cov(src_cov), dst_cov(dst_cov) {}

  /**
   * Factory to hide the construction of the CostFunction object from the client
   * code.
   */
  template <typename... Args>
  static ceres::CostFunction* Create(Args&&... args) {
    static constexpr const int kNumResidual = 3;
    static constexpr const int kNumRotParam = 4;
    static constexpr const int kNumTransParam = 3;
    return (
        new ceres::AutoDiffCostFunction<GICPCost, kNumResidual, kNumRotParam,
                                        kNumTransParam>(new GICPCost(args...)));
  }

  template <typename T>
  bool operator()(const T* const p_rxn, const T* const p_txn,
                  T* const p_err) const {
    // Unpack parameters.
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> txn{p_txn};
    const Eigen::Map<const Eigen::Quaternion<T>> q{p_rxn};
    const Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();

    // Compute residual.
    const Eigen::Matrix<T, 3, 1> delta =
        R * src.template cast<T>() + txn - dst.template cast<T>();

    // Compute combined covariance matrix.
    const Eigen::Matrix<T, 3, 3> cov =
        dst_cov.template cast<T>() +
        R * src_cov.template cast<T>() * R.transpose();

    // Compute generalized eigen-decomposition of covariance matrix.
    Eigen::EigenSolver<Eigen::Matrix<T, 3, 3>> eig(cov);

    // NOTE(yycho0108): The following code is invalid due to
    // undefined conversion between ceres::Jet<> and numeric literal types.
    // Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> eig(cov);

    // Compute inverse square-root of the covariance matrix.
    // {D = D'^{-1/2}, V = R} where C = R.D'.R
    const Eigen::Matrix<T, 3, 1> D = eig.eigenvalues().real().array().rsqrt();
    const Eigen::Matrix<T, 3, 3> V = eig.eigenvectors().real();
    const Eigen::Matrix<T, 3, 3> rsqrt_cov = V * D.asDiagonal() * V.transpose();

    // Finally, compute the residual term.
    Eigen::Map<Eigen::Matrix<T, 3, 1>>{p_err} = rsqrt_cov * delta;
    return true;
  }

  Eigen::Vector3f src;
  Eigen::Vector3f dst;
  RefMat3f src_cov;
  RefMat3f dst_cov;
};

}  // namespace rs_tracker
