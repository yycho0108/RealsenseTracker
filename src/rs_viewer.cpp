#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <librealsense2/hpp/rs_frame.hpp>
#include <thread>

#include <fmt/format.h>
#include <fmt/printf.h>
#include <Eigen/Core>
#include <librealsense2/rs.hpp>

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

void ConvertPointCloud(const rs2::points& cloud_in,
                       const rs2::video_frame& color_frame,
                       cho::core::PointCloud<float, 3>* const cloud_out,
                       std::vector<std::uint8_t>* const colors = nullptr) {
  // Hopefully nothing funny is going on with this.
  static_assert((sizeof(rs2::vertex) % sizeof(float)) == 0);

  const rs2::vertex* const vertices = cloud_in.get_vertices();
  const rs2::texture_coordinate* const tex_coords =
      cloud_in.get_texture_coordinates();

  if (cloud_in.size() > 0) {
    for (std::size_t i = 0; i < cloud_in.size(); ++i) {
      if (std::isnan(vertices[i].x) || std::isnan(vertices[i].y) ||
          std::isnan(vertices[i].z)) {
        continue;
      }
      if (std::abs(vertices[i].x) < 1e-8 && std::abs(vertices[i].y) < 1e-8 &&
          std::abs(vertices[i].z) < 1e-8) {
        continue;
      }
      fmt::print("v@{} = {} {} {} \n", i, vertices[i].x, vertices[i].y,
                 vertices[i].z);
      break;
    }
  }
#if 0
  cloud_out->GetData() =
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>,
                 Eigen::Unaligned,
                 Eigen::Stride<1, sizeof(rs2::vertex) / sizeof(float)>>(
          const_cast<float*>(reinterpret_cast<const float*>(vertices)),
          cloud_in.size(), 3)
          .transpose();
#else
  cloud_out->SetNumPoints(cloud_in.size());
  colors->resize(3 * cloud_in.size());

  const auto* const color_ptr =
      static_cast<const std::uint8_t*>(color_frame.get_data());
  const std::size_t stride = color_frame.get_stride_in_bytes();
  const int w = color_frame.get_width(), h = color_frame.get_height();
  for (std::size_t i = 0; i < cloud_in.size(); ++i) {
    cloud_out->GetPoint(i).x() =
        std::isnan(vertices[i].x) ? 0.0 : vertices[i].x;
    cloud_out->GetPoint(i).y() =
        std::isnan(vertices[i].y) ? 0.0 : vertices[i].y;
    cloud_out->GetPoint(i).z() =
        std::isnan(vertices[i].z) ? 0.0 : vertices[i].z;

    // fmt::print("{}x{}\n", tex_coords[i].u, tex_coords[i].v);
    const int x = std::min(
        std::max(static_cast<int>(tex_coords[i].u * w + .5f), 0), w - 1);
    const int y = std::min(
        std::max(static_cast<int>(tex_coords[i].v * h + .5f), 0), h - 1);
    const int offset = x * color_frame.get_bytes_per_pixel() + y * stride;
    if (0 <= offset && offset + 2 < color_frame.get_data_size()) {
      colors->at(i * 3) = color_ptr[offset];
      colors->at(i * 3 + 1) = color_ptr[offset + 1];
      colors->at(i * 3 + 2) = color_ptr[offset + 2];
    }
  }
#endif
}

struct RsViewer {
  RsViewer() : viewer{false} {}
  // RsViewer() : viewer{} {}
  void SetupPipeline() {
    fmt::print("Begin querying devices\n");
    for (const auto& dev : ctx.query_devices()) {
      rs2::pipeline pipe(ctx);
      rs2::config cfg;
      cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
      pipe.start(cfg);
      pipelines.emplace_back(std::move(pipe));
    }

    if (pipelines.empty()) {
      throw std::runtime_error(
          "No valid devices found! Failed to initialize pipelines.\n");
    }
  }
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

  void Loop() {
    fmt::print("Begin processing loop\n");
    double last_timestamp{0};
    while (true) {
      for (const auto& pipe : pipelines) {
        // Get input.
        auto frames = pipe.wait_for_frames();
        auto color_frame = frames.get_color_frame();
        auto depth_frame = frames.get_depth_frame();
        fmt::print("dstamp = {}\n", depth_frame.get_timestamp());
        if (depth_frame.get_timestamp() <= last_timestamp + 1000.f) {
          continue;
        }
        last_timestamp = depth_frame.get_timestamp();

        // Construct point cloud from color+depth_frame.
        cloud.map_to(color_frame);
        points = cloud.calculate(depth_frame);
        fmt::print("Got = {}\n", points.size());

        cho::core::PointCloud<float, 3> cloud_vis;
#if 1
        std::vector<std::uint8_t> colors;
        ConvertPointCloud(points, color_frame, &cloud_vis, &colors);
        // render_data.geometry = cloud_vis;
        if (cloud_vis.GetSize() >= 107) {
          fmt::print("{}\n", cloud_vis.GetPoint(107).transpose());
        }
#else
        cloud_vis.SetNumPoints(512);
        cloud_vis.GetData().setRandom();
#endif
        cho::vis::RenderData render_data{
            .tag = "cloud",
            .geometry = cloud_vis,
            .color = colors,
            .representation = cho::vis::RenderData::Representation::kPoints,
            .quit = false};
        viewer.Render(render_data);
      }
    }
  }

 private:
  rs2::pointcloud cloud;
  rs2::points points;
  rs2::context ctx;
  std::vector<rs2::pipeline> pipelines;

  cho::vis::SubprocessViewer viewer;
  // cho::vis::RemoteViewerClient viewer;
  // cho::vis::RenderData render_data;
};

int main() {
  RsViewer viewer;
  viewer.SetupPipeline();
  viewer.SetupViewer();
  viewer.Loop();
}
