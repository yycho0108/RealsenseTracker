#include "rs_tracker/data_source_rs.hpp"

namespace rs_tracker {

static void ConvertPointCloud(
    const rs2::points& cloud_in, const rs2::video_frame& color_frame,
    cho::core::PointCloud<float, 3>* const cloud_out,
    std::vector<std::uint8_t>* const colors = nullptr) {
  // Hopefully nothing funny is going on with this.
  static_assert((sizeof(rs2::vertex) % sizeof(float)) == 0);

  const rs2::vertex* const vertices = cloud_in.get_vertices();
  const rs2::texture_coordinate* const tex_coords =
      cloud_in.get_texture_coordinates();

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
  if (colors) {
    colors->resize(3 * cloud_in.size());
  }

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

    if (colors) {
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
  }
#endif
}

RealsenseSource::RealsenseSource(const double min_interval_ms)
    : min_interval_ms_(min_interval_ms) {
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

bool RealsenseSource::GetCloud(const double prev_timestamp,
                               cho::core::PointCloud<float, 3>* const cloud,
                               double* const curr_timestamp) {
  for (const auto& pipe : pipelines) {
    // Get input.
    auto frames = pipe.wait_for_frames();
    auto color_frame = frames.get_color_frame();
    auto depth_frame = frames.get_depth_frame();

    // Require that the point cloud is extracted at specified interval.
    if (depth_frame.get_timestamp() - prev_timestamp <= min_interval_ms_) {
      continue;
    }
    *curr_timestamp = depth_frame.get_timestamp();

    // Construct point cloud from color+depth_frame.
    proc_cloud.map_to(color_frame);
    points = proc_cloud.calculate(depth_frame);
    ConvertPointCloud(points, color_frame, cloud, nullptr);
    return true;
  }

  return false;
}

}  // namespace rs_tracker
