
#include <chrono>
#include <fstream>
#include <ios>
#include <iostream>
#include <thread>
#include <unordered_map>

#include <fmt/printf.h>
#include <glob.h>
#include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>

#include <cho_util/proto/geometry.pb.h>
#include <cho_util/proto/render.pb.h>
#include <cho_util/core/geometry/point_cloud.hpp>
#include <cho_util/type/convert.hpp>
#include <cho_util/vis/convert_proto.hpp>
#include <cho_util/vis/render_data.hpp>
#include <cho_util/vis/subprocess_viewer.hpp>

#include "rs_tracker/align/align_gicp.hpp"
#include "rs_tracker/align/align_icp.hpp"
#include "rs_tracker/align/teaser_interface.hpp"
#include "rs_tracker/common/fpfh.hpp"
#include "rs_tracker/common/point_cloud_utils.hpp"

namespace po = boost::program_options;

namespace rs_tracker {

struct RsReplayAppSettings {
  std::string record_file;
  float frame_interval_ms;
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
                    RsReplayAppSettings* const settings) {
  po::options_description desc("");

  // clang-format off
  desc.add_options()
      ("help,h", "help")
      ("record,r", po::value<std::string>(&settings->record_file)->default_value(""), "Input record file pattern, i.e. /tmp/{:04d}.pb")
      ("frame_interval,f", po::value<float>(&settings->frame_interval_ms)->default_value(1000.0f), "Frame interval, in milliseconds.")
      ;
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    Usage(desc);
    return false;
  }
  return true;
}

struct CloudAccumulator {
  struct VoxelHash {
    std::size_t operator()(const Eigen::Array3i& idx) const {
      std::size_t seed{0};
      boost::hash_combine(seed, idx.x());
      boost::hash_combine(seed, idx.y());
      boost::hash_combine(seed, idx.z());
      return seed;
    }
  };
  struct VoxelEqual {
    bool operator()(const Eigen::Array3i& lhs,
                    const Eigen::Array3i& rhs) const {
      return (lhs == rhs).all();
    }
  };
  explicit CloudAccumulator(const float voxel_size = 0.05f)
      : voxel_size_(voxel_size), voxel_size_inv_(1.0 / voxel_size_) {}

  void AddCloud(const Eigen::Isometry3f& xfm,
                const cho::core::PointCloud<float, 3>& cloud) {
    // Initialization...
    for (int i = 0; i < cloud.GetNumPoints(); ++i) {
      const Eigen::Vector3f p = xfm * cloud.GetPoint(i);
      const Eigen::Array3i idx = GetVoxelIndex(p);

      const auto it = voxel_map_.find(idx);
      if (it == voxel_map_.end()) {
        voxel_map_.emplace(idx, p);
      }
    }
  }

  Eigen::Array3i GetVoxelIndex(const Eigen::Vector3f& point) const {
    return (point.array() * voxel_size_inv_).cast<int>();
  }

  cho::core::PointCloud<float, 3> ExtractPointCloud() const {
    cho::core::PointCloud<float, 3> out;
    out.SetNumPoints(voxel_map_.size());
    int i{0};
    for (const auto& v : voxel_map_) {
      out.GetPoint(i) = v.second;
      ++i;
    }
    return out;
  }

 private:
  float voxel_size_;
  float voxel_size_inv_;
  std::unordered_map<Eigen::Array3i, Eigen::Vector3f, VoxelHash, VoxelEqual>
      voxel_map_;
};
}  // namespace rs_tracker

std::vector<std::string> Glob(const std::string& pattern) {
  // glob struct resides on the stack
  glob_t glob_result;
  memset(&glob_result, 0, sizeof(glob_result));

  // do the glob operation
  int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  if (return_value != 0) {
    globfree(&glob_result);
    std::stringstream ss;
    ss << "glob() failed with return_value " << return_value << std::endl;
    throw std::runtime_error(ss.str());
  }

  // collect all the filenames into a std::list<std::string>
  std::vector<std::string> filenames;
  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    filenames.push_back(std::string(glob_result.gl_pathv[i]));
  }

  // cleanup
  globfree(&glob_result);

  // done
  return filenames;
}

int main(int argc, char* argv[]) {
  rs_tracker::RsReplayAppSettings settings;
  if (!ParseArguments(argc, argv, &settings)) {
    return 1;
  }
  std::vector<std::string> record_files = Glob(settings.record_file);
  std::sort(record_files.begin(), record_files.end());

  // Instantiate viewer.
  cho::vis::SubprocessViewer viewer{false};
  viewer.Start();
  // std::this_thread::sleep_for(std::chrono::seconds(1));
  viewer.SetCameraPose(Eigen::Translation3f{0, 0, -5} *
                       Eigen::AngleAxisf(M_PI, Eigen::Vector3f{0, 1, 0}) *
                       Eigen::AngleAxisf(M_PI, Eigen::Vector3f{0, 0, 1}));
  int index{0};

  cho::core::PointCloud<float, 3> prev_cloud;
  Eigen::Isometry3f total_xfm = Eigen::Isometry3f::Identity();
  rs_tracker::CloudAccumulator acc{0.05};

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
  viewer.Render(render_data);

  while (1) {
    // Sanitize index against wrap-around
    if (index >= record_files.size()) {
      // index %= record_files.size();
      index = record_files.size() - 1;
    }

    // Read data.
    const std::string record_file = record_files.at(index);
    fmt::print("\t\tProcessing : {}\n", record_file);
    std::ifstream fin{record_file, std::ios_base::binary | std::ios_base::in};
    cho::proto::core::geometry::PointCloud cloud_proto;
    cloud_proto.ParseFromIstream(&fin);
    const auto& cloud_raw =
        cho::type::Convert<cho::core::PointCloud<float, 3>>(cloud_proto);

    // Remove Nans.
    cho::core::PointCloud<float, 3> cloud;
    rs_tracker::RemoveNans(cloud_raw, &cloud);

    // Sleep.
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int>(settings.frame_interval_ms)));

    Eigen::Isometry3f xfm = Eigen::Isometry3f::Identity();
    if (prev_cloud.IsEmpty()) {
      fmt::print("Init\n");
      // Initialize.
      prev_cloud = cloud;
      acc.AddCloud(total_xfm, prev_cloud);
    } else {
      // Register.
      fmt::print("Register\n");
#if 1
      cho::core::PointCloud<float, 3> curr_cloud_down, prev_cloud_down;
      rs_tracker::DownsampleVoxel(cloud, 0.05f, &curr_cloud_down);
      rs_tracker::DownsampleVoxel(prev_cloud, 0.05f, &prev_cloud_down);
      bool suc{false};
      if (1) {
        suc =
            rs_tracker::AlignIcp3d(curr_cloud_down, prev_cloud_down, 128, &xfm);
      } else if (0) {
        suc = rs_tracker::ComputeAlignment(curr_cloud_down, prev_cloud_down,
                                           &xfm);
      } else {
        rs_tracker::Cloud33f curr_fpfh, prev_fpfh;
        rs_tracker::ComputeFpfh(curr_cloud_down, Eigen::Vector3f{0, 0, 0}, 16,
                                0.25, &curr_fpfh);
        rs_tracker::ComputeFpfh(prev_cloud_down, Eigen::Vector3f{0, 0, 0}, 16,
                                0.25, &prev_fpfh);
        suc = rs_tracker::RegisterTeaser(
            curr_cloud_down.GetData(), prev_cloud_down.GetData(),
            curr_fpfh.GetData(), prev_fpfh.GetData(), 0.25f, &xfm);
      }

      if (suc) {
        total_xfm = total_xfm * xfm;
        acc.AddCloud(total_xfm, cloud);
        // acc.AddCloud(Eigen::Isometry3f::Identity(), cloud);
        prev_cloud = cloud;
      } else {
        fmt::print("ALIGNMENT FAILED!!\n");
      }
#else
      // Register directly to model
      const cho::core::PointCloud<float, 3>& total_cloud =
          acc.ExtractPointCloud();
      cho::core::PointCloud<float, 3> curr_cloud_down;
      rs_tracker::DownsampleVoxel(cloud, 0.05f, &curr_cloud_down);

      const bool suc =
          rs_tracker::AlignIcp3d(curr_cloud_down, total_cloud, &xfm);
      if (suc) {
        acc.AddCloud(xfm, cloud);
        prev_cloud = std::move(cloud);
      }
#endif

      if (false) {
        Eigen::MatrixXi matches;
        rs_tracker::Cloud33f curr_fpfh, prev_fpfh;
        rs_tracker::ComputeFpfh(curr_cloud_down, Eigen::Vector3f{0, 0, 0}, 16,
                                0.5f, &curr_fpfh);
        rs_tracker::ComputeFpfh(prev_cloud_down, Eigen::Vector3f{0, 0, 0}, 16,
                                0.5f, &prev_fpfh);
        rs_tracker::ComputeMatches(curr_fpfh, prev_fpfh, 2, &matches);
      }
    }

    if (false) {
      cho::vis::RenderData render_data{
          "prev",
          prev_cloud,
          {255, 0, 0},
          1.0f,
          cho::vis::RenderData::Representation::kPoints,
          false};
      viewer.Render(render_data);
    }

    if (false) {
      cho::vis::RenderData render_data{
          "curr",
          cloud,
          {0, 255, 0},
          1.0f,
          cho::vis::RenderData::Representation::kPoints,
          false};
      viewer.Render(render_data);
    }

    if (false) {
      cho::core::PointCloud<float, 3> cloud_transformed;
      cloud_transformed.SetNumPoints(cloud.GetNumPoints());
      for (int i = 0; i < cloud.GetNumPoints(); ++i) {
        cloud_transformed.GetPoint(i) = xfm.inverse() * cloud.GetPoint(i);
      }

      cho::vis::RenderData render_data{
          "align",
          cloud_transformed,
          {0, 0, 255},
          1.0f,
          cho::vis::RenderData::Representation::kPoints,
          false};
      viewer.Render(render_data);
    }

    if (true) {
      cho::core::PointCloud<float, 3> cloud_transformed;
      cloud_transformed.SetNumPoints(cloud.GetNumPoints());
      for (int i = 0; i < cloud.GetNumPoints(); ++i) {
        cloud_transformed.GetPoint(i) = total_xfm * cloud.GetPoint(i);
      }
      cho::vis::RenderData render_data{
          "cloud",
          cloud_transformed,
          {255, 0, 0},
          1.0f,
          cho::vis::RenderData::Representation::kPoints,
          false};
      viewer.Render(render_data);
    }

    if (true) {
      cho::vis::RenderData render_data{
          "model",
          acc.ExtractPointCloud(),
          {255, 255, 255},
          1.0f,
          cho::vis::RenderData::Representation::kPoints,
          false};
      viewer.Render(render_data);
    }

    if (false) {
      fmt::print("SHOW NORMALS ...\n");
      rs_tracker::Cloud3f cloud_transformed;

      fmt::print("DOWN\n");
      rs_tracker::DownsampleVoxel(cloud, 0.05f, &cloud_transformed);

      const Eigen::Vector3f viewpoint = total_xfm * Eigen::Vector3f{0, 0, 0};

      fmt::print("XFM\n");
      for (int i = 0; i < cloud_transformed.GetNumPoints(); ++i) {
        cloud_transformed.GetPoint(i) =
            total_xfm * cloud_transformed.GetPoint(i);
      }

      fmt::print("TREE\n");
      rs_tracker::KDTree3f tree{std::cref(cloud_transformed), 16};

      fmt::print("COMPUTE\n");
      rs_tracker::Cloud3f normals;
      rs_tracker::ComputeNormals(cloud_transformed, tree, 16, &normals);
      rs_tracker::OrientNormals(cloud_transformed, viewpoint, &normals);

      fmt::print("FORMAT\n");
      cho::core::Lines<float, 3> normal_vis;
      normal_vis.SetNumLines(normals.GetNumPoints());
      for (int i = 0; i < normals.GetNumPoints(); ++i) {
        normal_vis.GetSourcePoint(i) = cloud_transformed.GetPoint(i);
        normal_vis.GetTargetPoint(i) =
            cloud_transformed.GetPoint(i) + 0.2f * normals.GetPoint(i);
      }
      fmt::print("STRUCT\n");
      cho::vis::RenderData render_data{
          "normals",
          normal_vis,
          {},
          1.0f,
          cho::vis::RenderData::Representation::kWireframe,
          false};
      fmt::print("RENDER\n");
      viewer.Render(render_data);
      fmt::print("SHOWED NORMALS ...\n");
    }

    // Increment index.
    ++index;
  }

  return 0;
}
