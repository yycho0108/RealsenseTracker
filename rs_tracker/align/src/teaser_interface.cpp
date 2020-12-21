#include "rs_tracker/align/teaser_interface.hpp"

#include <fstream>

#include "teaser/fpfh.h"
#include "teaser/matcher.h"
#include "teaser/registration.h"

namespace rs_tracker {

/**
 * @brief
 *
 * @param src_cloud_vv
 * @param dst_cloud_vv
 * @param transform
 *
 * @return
 */
bool RegisterTeaser(const Eigen::Matrix<float, 3, Eigen::Dynamic>& src_cloud_vv,
                    const Eigen::Matrix<float, 3, Eigen::Dynamic>& dst_cloud_vv,
                    const Eigen::Matrix<float, 33, Eigen::Dynamic>& src_fpfh,
                    const Eigen::Matrix<float, 33, Eigen::Dynamic>& dst_fpfh,
                    const float noise_bound,
                    Eigen::Isometry3f* const transform) {
  // Convert input to teaser point cloud.
  teaser::PointCloud src_cloud;
  teaser::PointCloud tgt_cloud;
  teaser::FPFHCloud src_feats;
  teaser::FPFHCloud tgt_feats;
  for (int i = 0; i < src_cloud_vv.cols(); ++i) {
    const auto& point = src_cloud_vv.col(i);
    const auto& fpfh = src_fpfh.col(i);

    pcl::FPFHSignature33 fpfh_pcl;
    std::copy(fpfh.data(), fpfh.data() + 33, fpfh_pcl.histogram);

    src_cloud.push_back({point.x(), point.y(), point.z()});
    src_feats.push_back(fpfh_pcl);
  }
  for (int i = 0; i < dst_cloud_vv.cols(); ++i) {
    const auto& point = dst_cloud_vv.col(i);
    const auto& fpfh = dst_fpfh.col(i);

    pcl::FPFHSignature33 fpfh_pcl;
    std::copy(fpfh.data(), fpfh.data() + 33, fpfh_pcl.histogram);
    tgt_cloud.push_back({point.x(), point.y(), point.z()});
    tgt_feats.push_back(fpfh_pcl);
  }

  // Compute FPFH features.
  // teaser::FPFHEstimation fpfh;
  // auto src_feats =
  //     fpfh.computeFPFHFeatures(src_cloud, viewpoint.x(), viewpoint.y(),
  //                              viewpoint.z(), normal_radius, feature_radius);
  // auto tgt_feats =
  //     fpfh.computeFPFHFeatures(tgt_cloud, viewpoint.x(), viewpoint.y(),
  //                              viewpoint.z(), normal_radius, feature_radius);

  // const auto& h = src_feats->front().histogram;
  // OINFO("FPFH(PCL)#0 = {}", fmt::join(std::begin(h), std::end(h), " "));
  // std::ofstream fout("/tmp/histo_pcl.txt");
  // fmt::print(fout, "{}", fmt::join(std::begin(h), std::end(h), " "));

  // Compute correspondences.
  teaser::Matcher matcher;
  auto correspondences = matcher.calculateCorrespondences(
      src_cloud, tgt_cloud, src_feats, tgt_feats, false, true, false, 0.95);

  // At least 3 correspondences required (I think?) for 3DoF estimation.
  if (correspondences.size() <= 3) {
    return false;
  }

  // OINFO("Sizes ... {} / ({},{})\n", correspondences.size(), src_cloud.size(),
  // tgt_cloud.size());

  // Prepare solver parameters
  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = noise_bound;
  params.cbar2 = 1;
  params.estimate_scaling = false;
  params.rotation_max_iterations = 2048;
  // params.rotation_gnc_factor = 1.4;
  params.rotation_gnc_factor = 1.4;

  params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  // params.rotation_estimation_algorithm =
  //   teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
  params.rotation_cost_threshold = 1e-6;
  params.inlier_selection_mode =
      teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT;
  // params.inlier_selection_mode =
  // teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::NONE;
  params.rotation_tim_graph =
      teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::COMPLETE;

  params.kcore_heuristic_threshold = 0.5;  // skip max clique
  // params.kcore_heuristic_threshold = 0.0;  // skip max clique
  // params.use_max_clique = false;

  // Solve with TEASER++
  teaser::RobustRegistrationSolver solver(params);
  solver.solve(src_cloud, tgt_cloud, correspondences);
  auto solution = solver.getSolution();

  // OINFO("R = {}", solution.rotation);
  // OINFO("T = {}", solution.translation);

  // Compose transform.
  *transform = Eigen::Translation3f{solution.translation.cast<float>()} *
               Eigen::Quaternionf{solution.rotation.cast<float>()};

#if 0
  // Export input/output for validation.
  if (solution.valid) {
    std::ofstream fout_src{"/tmp/src_cloud.txt"};
    for (const auto& x : src_cloud_vv) {
      fout_src << x.transpose() << std::endl;
    }
    std::ofstream fout_dst{"/tmp/dst_cloud.txt"};
    for (const auto& x : dst_cloud_vv) {
      fout_dst << x.transpose() << std::endl;
    }
    std::ofstream fout_xfm{"/tmp/transform.txt"};
    fout_xfm << solution.rotation << std::endl;
    fout_xfm << solution.translation.transpose() << std::endl;
  }
#endif

  return solution.valid;
}
}  // namespace rs_tracker
