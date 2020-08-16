#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <thread>

#include <fmt/format.h>
#include <fmt/printf.h>
#include <Eigen/Core>

#include <cho_util/proto/render.pb.h>
#include <cho_util/core/geometry.hpp>
#include <cho_util/core/geometry/line.hpp>
#include <cho_util/core/geometry/point_cloud.hpp>
#include <cho_util/core/geometry/sphere.hpp>
#include <cho_util/type/convert.hpp>

// #include <cho_util/vis/direct_viewer.hpp>
#include <cho_util/vis/render_data.hpp>
#include <cho_util/vis/subprocess_viewer.hpp>
// #include <cho_util/vis/remote_viewer.hpp>

#include "rs_tracker/align.hpp"
#include "rs_tracker/data_source.hpp"
#include "rs_tracker/data_source_rs.hpp"
#include "rs_tracker/point_cloud_utils.hpp"

struct RsTracker {
  RsTracker() : viewer{false}, source_{} {}
  // RsTracker() : viewer{} {}

  void SetupViewer() {
    viewer.Start();

    //;cho::core::PointCloud<float, 3> cloud_geom;
    //;render_data = cho::vis::RenderData{
    //;    .tag = "cloud",
    //;    .geometry = cloud_geom,
    //;    .color = {255, 255, 255},
    //;    .representation = cho::vis::RenderData::Representation::kPoints,
    //;    .quit = false};
  }

  void Render() {
    if (curr_cloud.GetNumPoints() <= 0) {
      // warn: attempting to render invalid point cloud.
      return;
    }

    const cho::vis::RenderData render_data{
        .tag = "cloud",
        .geometry = curr_cloud,
        .color = {255, 255, 255},
        .representation = cho::vis::RenderData::Representation::kPoints,
        .quit = false};
    viewer.Render(std::move(render_data));
  }

  void Loop() {
    fmt::print("Begin processing loop\n");
    double proc_timestamp{0};
    while (true) {
      // Retrieve data.
      const bool has_cloud =
          source_.GetCloud(proc_timestamp, &curr_cloud, &proc_timestamp);
      if (!has_cloud) {
        continue;
      }

      if (true) {
        rs_tracker::DownsampleVoxel(curr_cloud, 0.1, &curr_cloud);
      }

      // Initialize & attempt alignment if both point clouds are available.
      if (prev_cloud.GetNumPoints() > 0 && curr_cloud.GetNumPoints() > 0) {
        fmt::print("Align {} - {}\n", prev_cloud.GetNumPoints(),
                   curr_cloud.GetNumPoints());
        Eigen::Isometry3f transform;
        rs_tracker::ComputeAlignment(prev_cloud.GetData().transpose(),
                                     curr_cloud.GetData().transpose(),
                                     &transform);
        fmt::print("xfm={}\n", transform);
        fmt::print("DONE\n");
      }

      // Visualize ...
      Render();

      // Prepare for the next step.
      // Hopefully, this takes care of move.
      prev_cloud = std::move(curr_cloud);
    }
  }

 private:
  cho::vis::SubprocessViewer viewer;
  cho::core::PointCloud<float, 3> prev_cloud;
  cho::core::PointCloud<float, 3> curr_cloud;
  // cho::vis::RemoteViewerClient viewer;
  // cho::vis::RenderData render_data;

  // rs_tracker::RandomSource source_;
  rs_tracker::RealsenseSource source_;
};

int main() {
  RsTracker tracker;
  tracker.SetupViewer();
  tracker.Loop();
}
