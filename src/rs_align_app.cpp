#include <fstream>
#include <iostream>

#include <fmt/printf.h>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <boost/program_options.hpp>

#include <cho_util/type/convert.hpp>
#include <cho_util/vis/convert_proto.hpp>
#include <cho_util/vis/subprocess_viewer.hpp>

#include "teaser/interface.hpp"

#include "rs_tracker/align_icp.hpp"
#include "rs_tracker/common.hpp"
#include "rs_tracker/fpfh.hpp"
#include "rs_tracker/point_cloud_utils.hpp"

namespace po = boost::program_options;

struct RsAlignAppSettings {
  std::string source_file{"../data/0020.pb"};
  std::string target_file{"../data/0021.pb"};
  float voxel_size{0.05F};
  int normal_k{16};
  float feature_radius{0.5F};
  float lowe_ratio{0.9F};
  bool init_with_fpfh{true};
  bool refine_with_icp{true};
  bool use_teaser{true};
};

void Usage(const po::options_description& desc) {
  // clang-format off
  fmt::print(R"(Usage:
  rs_replay_app [options]
Description:
  Replay app for realsense record.
Options:
{}
  )",
             desc);
  // clang-format on
}

bool ParseArguments(int argc, char* argv[],
                    RsAlignAppSettings* const settings) {
  po::options_description desc("");

#define ADD_SETTINGS(x, y, z)                        \
  ((std::string(#x) + "," + std::string(y)).c_str(), \
   po::value(&settings->x)->default_value(settings->x), z)

  // clang-format off
  desc.add_options()
      ("help,h", "help")
      ADD_SETTINGS(source_file, "s", "Source cloud to transform")
      ADD_SETTINGS(target_file, "t", "Target cloud to align to")
      ADD_SETTINGS(voxel_size, "v", "Voxel size")
      ADD_SETTINGS(normal_k, "k", "Num nearest neihbors for normals")
      ADD_SETTINGS(feature_radius, "r", "Feature radius for FPFH")
      ADD_SETTINGS(lowe_ratio, "l", "Lowe ratio")
      ADD_SETTINGS(init_with_fpfh, "i", "Initialize corasely with fpfh feature matching.")
      ADD_SETTINGS(refine_with_icp, "x", "Refine with ICP")
      ADD_SETTINGS(use_teaser, "q", "use teaser")
      ;
  // clang-format on
#undef ADD_SETTINGS

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    Usage(desc);
    return false;
  }
  return true;
}

static rs_tracker::Cloud3f ReadCloud(const std::string& filename) {
  std::ifstream fin{filename, std::ios_base::binary | std::ios_base::in};
  cho::proto::core::geometry::PointCloud cloud_proto;
  cloud_proto.ParseFromIstream(&fin);
  const auto& cloud_raw = cho::type::Convert<rs_tracker::Cloud3f>(cloud_proto);
  return cloud_raw;
}
using Vector33f = Eigen::Matrix<float, 33, 1>;

void ComputePCAProjection(const rs_tracker::Cloud33f& fpfh,
                          Vector33f* const center,
                          Eigen::Matrix<float, 3, 33>* const projection) {
  const int n = fpfh.GetNumPoints();
  *center = fpfh.GetData().rowwise().mean();

  // Center data.
  const Eigen::Matrix<float, 33, Eigen::Dynamic> centered =
      fpfh.GetData().colwise() - *center;

  // Compute SVD.
  Eigen::JacobiSVD<Eigen::Matrix<float, 33, Eigen::Dynamic>> svd{
      centered, Eigen::ComputeThinU};

  // Extract scale component ...
  const Eigen::Array3f scale =
      (svd.singularValues().head<3>().array().inverse() *
       static_cast<float>(std::sqrt(n - 1.0F)));

  // Output full linear projection matrix.
  *projection =
      (svd.matrixU().leftCols<3>().array().rowwise() * scale.transpose())
          .transpose();
}

void ApplyPCAProjection(const rs_tracker::Cloud33f& fpfh,
                        const Vector33f& center,
                        const Eigen::Matrix<float, 3, 33>& projection,
                        rs_tracker::Cloud3f* const out) {
  out->GetData() = projection * (fpfh.GetData().colwise() - center);
}

void ColorizeFpfh(const rs_tracker::Cloud33f& fpfh,
                  rs_tracker::Cloud3f* const colors) {
  // Center data.
  const Eigen::Matrix<float, 33, Eigen::Dynamic> centered =
      fpfh.GetData().colwise() - fpfh.GetData().rowwise().mean();
  Eigen::JacobiSVD<Eigen::Matrix<float, 33, Eigen::Dynamic>> svd{
      centered, Eigen::ComputeThinV};
  // project.
  colors->GetData() = svd.matrixV().leftCols<3>().transpose();
  // whiten.
  colors->GetData() *= std::sqrt(fpfh.GetNumPoints() - 1);
}

void DrawAxis(cho::vis::SubprocessViewer& v) {
  // Show Axes
  cho::core::Lines<float, 3> axes_lines;
  axes_lines.SetNumLines(3);
  axes_lines.GetSourcePoint(0).setZero();
  axes_lines.GetSourcePoint(1).setZero();
  axes_lines.GetSourcePoint(2).setZero();
  axes_lines.GetTargetPoint(0) = Eigen::Vector3f{0.1, 0, 0};
  axes_lines.GetTargetPoint(1) = Eigen::Vector3f{0, 0.1, 0};
  axes_lines.GetTargetPoint(2) = Eigen::Vector3f{0, 0, 0.1};
  cho::vis::RenderData render_data{
      "axis",
      axes_lines,
      {
          255, 0, 0,
          // 255,
          // 0,
          // 0,
          0, 255, 0,
          // 0,
          // 255,
          // 0,
          0, 0, 255,
          // 0,
          // 0,
          // 255,
      },
      1.0f,
      cho::vis::RenderData::Representation::kWireframe,
      false};
  v.Render(render_data);
}

void DrawCloud(cho::vis::SubprocessViewer& v, const std::string& name,
               const rs_tracker::Cloud3f& cloud, const Eigen::Vector3i& color) {
  std::vector<std::uint8_t> colors{color.data(), color.data() + 3};
  cho::vis::RenderData render_data{
      name, cloud, colors, 1.0f, cho::vis::RenderData::Representation::kPoints,
      false};
  v.Render(render_data);
}

std::vector<std::pair<int, int>> PruneMatchesLowe(
    const Eigen::MatrixXi& matches, const rs_tracker::Cloud33f& src_fpfh,
    const rs_tracker::Cloud33f& dst_fpfh, const float lowe_ratio,
    std::vector<float>* const weights = nullptr) {
  using Vector33f = Eigen::Matrix<float, 33, 1>;
  std::vector<std::pair<int, int>> indices;
  indices.reserve(matches.rows());
  weights->clear();
  weights->reserve(matches.rows());
  for (int i = 0; i < matches.rows(); ++i) {
    const int j0 = matches.coeffRef(i, 0);
    const int j1 = matches.coeffRef(i, 1);

    // Get features
    const Vector33f& p_src = src_fpfh.GetPoint(i);
    const Vector33f& p_dst0 = dst_fpfh.GetPoint(j0);
    const Vector33f& p_dst1 = dst_fpfh.GetPoint(j1);

    // Recompute feature-space distance
    const float d0 = (p_src - p_dst0).squaredNorm();
    const float d1 = (p_src - p_dst1).squaredNorm();

    static constexpr const float kVar{0.25 * 0.25};
    if (d0 < d1) {
      if (d0 < lowe_ratio * d1) {
        indices.emplace_back(i, j0);
        if (weights) {
          weights->emplace_back(std::exp(-d0 / kVar));
        }
      }
    } else {
      if (d1 < lowe_ratio * d0) {
        indices.emplace_back(i, j1);
        if (weights) {
          weights->emplace_back(std::exp(-d1 / kVar));
        }
      }
    }
  }
  return indices;
}

void DrawMatches(cho::vis::SubprocessViewer& v, const std::string& name,
                 const rs_tracker::Cloud3f& src_cloud,
                 const rs_tracker::Cloud3f& dst_cloud,
                 const std::vector<std::pair<int, int>>& indices,
                 const Eigen::Vector3i& color) {
  std::vector<std::uint8_t> colors{color.data(), color.data() + 3};
  cho::core::Lines<float, 3> lines;
  lines.SetNumLines(indices.size());
  for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
    const auto& index = indices[i];
    lines.GetSourcePoint(i) = src_cloud.GetPoint(index.first);
    lines.GetTargetPoint(i) = dst_cloud.GetPoint(index.second);
  }

  cho::vis::RenderData render_data{
      name,
      lines,
      colors,
      1.0f,
      cho::vis::RenderData::Representation::kWireframe,
      false};
  v.Render(render_data);
}

int main(int argc, char* argv[]) {
  // Parse settings.
  RsAlignAppSettings settings;
  if (!ParseArguments(argc, argv, &settings)) {
    return 1;
  }

  // Parse inputs.
  const rs_tracker::Cloud3f src_cloud_raw = ReadCloud(settings.source_file);
  const rs_tracker::Cloud3f dst_cloud_raw = ReadCloud(settings.target_file);
  rs_tracker::Cloud3f src_cloud, dst_cloud;
  rs_tracker::DownsampleVoxel(src_cloud_raw, settings.voxel_size, &src_cloud);
  rs_tracker::DownsampleVoxel(dst_cloud_raw, settings.voxel_size, &dst_cloud);

#if 0
  const Eigen::Quaternionf R =
      Eigen::AngleAxisf{0.1, Eigen::Vector3f{1, 0, 0}} *
      Eigen::AngleAxisf{-0.2, Eigen::Vector3f{0, 1, 0}} *
      Eigen::AngleAxisf{0.25, Eigen::Vector3f{0, 0, 1}};
  dst_cloud.GetData() = R.toRotationMatrix() * dst_cloud.GetData();
#endif

  // Setup viewer.
  cho::vis::SubprocessViewer viewer{false};
  viewer.Start();
  viewer.SetCameraPose(Eigen::Translation3f{0, 0, -5} *
                       Eigen::AngleAxisf(M_PI, Eigen::Vector3f{0, 1, 0}) *
                       Eigen::AngleAxisf(M_PI, Eigen::Vector3f{0, 0, 1}));

  // Compute correspondences based on FPFH.
  rs_tracker::Cloud33f src_fpfh, dst_fpfh;
  Eigen::MatrixXi matches;
  rs_tracker::ComputeFpfh(src_cloud, Eigen::Vector3f{0, 0, 0},
                          settings.normal_k, settings.feature_radius,
                          &src_fpfh);
  rs_tracker::ComputeFpfh(dst_cloud, Eigen::Vector3f{0, 0, 0},
                          settings.normal_k, settings.feature_radius,
                          &dst_fpfh);
  rs_tracker::ComputeMatches(src_fpfh, dst_fpfh, 2, &matches);
  fmt::print("FPFH_MAX : {}\n",
             src_fpfh.GetData().rowwise().maxCoeff().transpose());

  std::vector<float> weights;
  const auto& indices = PruneMatchesLowe(matches, src_fpfh, dst_fpfh,
                                         settings.lowe_ratio, &weights);
  fmt::print("MATCH : {} / {}\n", indices.size(), matches.rows());

  // Compute transform ...
  Eigen::Isometry3f xfm = Eigen::Isometry3f::Identity();
  if (settings.init_with_fpfh) {
    fmt::print("KABSCH!\n");
    const bool suc =
        rs_tracker::SolveKabsch(src_cloud, dst_cloud, indices, weights, &xfm);
    fmt::print("kasbch matrix : {}\n", xfm.matrix());
    if (!suc) {
      fmt::print("kabsch failed\n");
    }
  }
  if (settings.refine_with_icp) {
    fmt::print("ICP!\n");
    const bool suc = rs_tracker::AlignIcp3d(src_cloud, dst_cloud, 128, &xfm);
    fmt::print("icp matrix : {}\n", xfm.matrix());
    if (!suc) {
      fmt::print("icp failed\n");
    }
  }

  if (settings.use_teaser) {
    rs_tracker::RegisterTeaser(src_cloud.GetData(), dst_cloud.GetData(),
                               src_fpfh.GetData(), dst_fpfh.GetData(), 0.25f,
                               &xfm);
  }

  // Apply transform ...
  rs_tracker::Cloud3f rec_cloud;
  rec_cloud.GetData() = xfm * src_cloud.GetData();

  DrawAxis(viewer);
  // DrawCloud(viewer, "source", src_cloud,
  //          Eigen::Vector3i{255, 0, 0});  // src = red
  // DrawCloud(viewer, "target", dst_cloud,
  // Eigen::Vector3i{0, 255, 0});  // dst = green
  // DrawCloud(viewer, "rec", rec_cloud,
  // Eigen::Vector3i{0, 0, 255});  // rec = blue
  // DrawMatches(viewer, "matches", rec_cloud, dst_cloud, indices,
  // Eigen::Vector3i{0, 0, 255});

  // draw_fpfh
  if (true) {
    Vector33f pc;
    Eigen::Matrix<float, 3, 33> pp;
    ComputePCAProjection(src_fpfh, &pc, &pp);
    // ColorizeFpfh(src_fpfh, &src_fpfh_colors);

    rs_tracker::Cloud3f src_fpfh_colors;
    ApplyPCAProjection(src_fpfh, pc, pp, &src_fpfh_colors);
    // for (int i = 0; i < src_fpfh.GetNumPoints(); ++i) {
    //  fmt::print("{} vs {}\n", src_fpfh_colors.GetPoint(i).transpose(),
    //             src_fpfh_colors_v2.GetPoint(i).transpose());
    //}
    std::vector<std::uint8_t> src_colors_vec(src_fpfh_colors.GetNumPoints() *
                                             3);
    for (int i = 0; i < src_fpfh_colors.GetNumPoints(); ++i) {
      const auto& scol = src_fpfh_colors.GetPoint(i);
      src_colors_vec[i * 3] =
          std::max(0.0f, std::min(255.0f, 255.0f * (scol.x() + 2) / 4));
      src_colors_vec[i * 3 + 1] =
          std::max(0.0f, std::min(255.0f, 255.0f * (scol.y() + 2) / 4));
      src_colors_vec[i * 3 + 2] =
          std::max(0.0f, std::min(255.0f, 255.0f * (scol.z() + 2) / 4));
    }

    cho::vis::RenderData render_data{
        "src_fpfh",
        rec_cloud,
        src_colors_vec,
        1.0f,
        cho::vis::RenderData::Representation::kPoints,
        false};
    viewer.Render(render_data);

    // Show target colorized fpfh.
    rs_tracker::Cloud3f dst_fpfh_colors;
    ApplyPCAProjection(dst_fpfh, pc, pp, &dst_fpfh_colors);
    std::vector<std::uint8_t> dst_colors_vec(dst_fpfh_colors.GetNumPoints() *
                                             3);
    for (int i = 0; i < dst_fpfh_colors.GetNumPoints(); ++i) {
      const auto& scol = dst_fpfh_colors.GetPoint(i);
      dst_colors_vec[i * 3] =
          std::max(0.0f, std::min(255.0f, 255.0f * (scol.x() + 2) / 4));
      dst_colors_vec[i * 3 + 1] =
          std::max(0.0f, std::min(255.0f, 255.0f * (scol.y() + 2) / 4));
      dst_colors_vec[i * 3 + 2] =
          std::max(0.0f, std::min(255.0f, 255.0f * (scol.z() + 2) / 4));
    }

    cho::vis::RenderData dst_render_data{
        "dst_fpfh",
        dst_cloud,
        dst_colors_vec,
        1.0f,
        cho::vis::RenderData::Representation::kPoints,
        false};
    viewer.Render(dst_render_data);
  }
  viewer.Spin();
}
