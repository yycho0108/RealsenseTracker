#include "rs_tracker/align.hpp"

#include <fmt/printf.h>

#include "rs_tracker/common.hpp"
#include "rs_tracker/gicp_cost.hpp"
#include "rs_tracker/point_cloud_utils.hpp"

namespace rs_tracker {

// TODO(yycho0108): Consider exposing options interface.
static ceres::Solver::Options GetOptions() {
  // Set a few options
  ceres::Solver::Options options;
  // options.use_nonmonotonic_steps = true;
  // options.preconditioner_type = ceres::IDENTITY;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 1024;
  options.trust_region_strategy_type =
      ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  // options.minimizer_type = ceres::MinimizerType::LINE_SEARCH;
  // options.line_search_direction_type = ceres::LineSearchDirectionType::BFGS;

  // options.minimizer_progress_to_stdout = true;

  //    options.preconditioner_type = ceres::SCHUR_JACOBI;
  //    options.linear_solver_type = ceres::DENSE_SCHUR;
  //    options.use_explicit_schur_complement=true;
  //    options.max_num_iterations = 100;

  // cout << "Ceres Solver getOptions()" << endl;
  // cout << "Ceres preconditioner type: " << options.preconditioner_type <<
  // endl; cout << "Ceres linear algebra type: " <<
  // options.sparse_linear_algebra_library_type << endl; cout << "Ceres linear
  // solver type: " << options.linear_solver_type << endl;

  return options;
}

float ComputeAlignment(const Cloud& src, const Cloud& dst,
                       const std::vector<Eigen::Matrix3f>& src_covs,
                       const std::vector<Eigen::Matrix3f>& dst_covs,
                       const std::vector<int>& dst_indices,
                       const Eigen::Isometry3f& seed,
                       Eigen::Isometry3f* transform) {
  // Convert Seed -> param.
  const Eigen::Matrix3d& R{seed.linear().cast<double>()};

  // Initialize quaternion from rotation matrix and retrieve data pointer.
  Eigen::Quaterniond q{R};
  double* const rxn = q.coeffs().data();

  double txn[3];
  Eigen::Map<Eigen::Matrix<double, 3, 1>>{txn} =
      seed.translation().cast<double>();

  // Define problem.
  ceres::Problem problem;
  for (int i = 0; i < dst_indices.size(); ++i) {
    const int src_i = i;
    const int dst_i = dst_indices[i];

    ceres::CostFunction* cost =
        GICPCost::Create(src.row(src_i).transpose(), dst.row(dst_i).transpose(),
                         src_covs[src_i], dst_covs[dst_i]);
    ceres::LossFunction* loss = new ceres::SoftLOneLoss(2.0f);
    // ceres::LossFunction* loss = nullptr;
    problem.AddResidualBlock(cost, loss, rxn, txn);
  }

  auto p = new ceres::EigenQuaternionParameterization;
  problem.SetParameterization(rxn, p);

  // Solve problem.
  ceres::Solver::Summary summary;
  ceres::Solve(GetOptions(), &problem, &summary);

#if 0
  // Covariance?
  ceres::Covariance::Options cov_opts;
  cov_opts.algorithm_type = ceres::DENSE_SVD;
  cov_opts.null_space_rank = -1;
  ceres::Covariance cov_alg{cov_opts};
  std::vector<std::pair<const double*, const double*>> cov_blocks;
  cov_blocks.emplace_back(txn, txn);
  cov_blocks.emplace_back(rxn, rxn);
  cov_alg.Compute(cov_blocks, &problem);
  double cov_txn[3 * 3];
  double cov_rxn[3 * 3];
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> cov_txn_mat{cov_txn};
  fmt::print("{}\n", cov_txn_mat);
  cov_alg.GetCovarianceBlock(txn, txn, cov_txn);
  cov_alg.GetCovarianceBlockInTangentSpace(rxn, rxn, cov_rxn);
#endif

  // Convert param -> transform.
  const Eigen::Vector3d t_out = Eigen::Map<Eigen::Vector3d>{txn};
  const Eigen::Matrix3d R_out = q.toRotationMatrix();
  transform->translation() = t_out.cast<float>();
  transform->linear() = R_out.cast<float>();
  return summary.final_cost;
}

float ComputeAlignment(const Cloud& src, const Cloud& dst,
                       Eigen::Isometry3f* const transform) {
  fmt::print("{}x{}, {}x{}", src.rows(), src.cols(), dst.rows(), dst.cols());
  static constexpr const int kMaxIter = 128;

  // Correspondences
  fmt::print("TREE\n");
  std::shared_ptr<KDTree> src_tree =
      std::make_shared<KDTree>(3, std::cref(src), 10);
  src_tree->index->buildIndex();
  std::shared_ptr<KDTree> dst_tree =
      std::make_shared<KDTree>(3, std::cref(dst), 10);
  dst_tree->index->buildIndex();

  // Covariances
  fmt::print("COV\n");
  std::vector<Eigen::Matrix3f> src_covs(src.rows(),
                                        Eigen::Matrix3f::Identity());
  ComputeCovariances(*src_tree, src, &src_covs, true);
  std::vector<Eigen::Matrix3f> dst_covs(dst.rows(),
                                        Eigen::Matrix3f::Identity());
  ComputeCovariances(*dst_tree, dst, &dst_covs, true);

  // TODO(yycho0108): Accept estimate as an input argument.
  // Eigen::Isometry3f estimate = transform;
  Eigen::Isometry3f estimate = Eigen::Isometry3f::Identity();
  Cloud tmp = (estimate * src.transpose()).transpose();
  std::vector<Eigen::Matrix3f> tmp_covs(src_covs);
  float cost{0};
  for (int i = 0; i < kMaxIter; ++i) {
    // fmt::print("\r{}/{}", i, kMaxIter);

    // Determine correspondences.
    std::vector<int> nn_indices;
    std::vector<float> nn_sq_dists;
    FindCorrespondences(*dst_tree, tmp, &nn_indices, &nn_sq_dists);

    // Compute Alignment.
    Eigen::Isometry3f delta_xfm = Eigen::Isometry3f::Identity();
    cost = ComputeAlignment(src, dst, src_covs, dst_covs, nn_indices, estimate,
                            &delta_xfm);

    // Update transforms.
    estimate = delta_xfm;
    tmp.transpose() = estimate * src.transpose();
  }
  *transform = estimate;
  fmt::print("final cost : {}\n", cost);
  return cost;
}

}  // namespace rs_tracker
