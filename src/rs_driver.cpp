#include "rs_tracker/rs_driver.hpp"

#include <algorithm>
#include <shared_mutex>
#include <thread>

#include <fmt/format.h>
#include <fmt/printf.h>

#include <librealsense2/hpp/rs_frame.hpp>
#include <librealsense2/rs.hpp>

#include <cho_util/core/geometry/point_cloud.hpp>

namespace rs_tracker {

void ConvertImage(const rs2::video_frame& frame, const int type,
                  cv::Mat* const image) {
  const std::size_t stride = frame.get_stride_in_bytes();
  const int w = frame.get_width(), h = frame.get_height();
  const auto* const frame_ptr =
      static_cast<const std::uint8_t*>(frame.get_data());

  // Allocate output.
  if (image->size() != cv::Size(w, h) || image->type() != type) {
    image->create(h, w, type);
  }

  // Copy actual data.
  std::copy(frame_ptr, frame_ptr + frame.get_data_size(), image->data);
}

void ConvertPointCloud(
    const rs2::points& cloud_in, const rs2::video_frame& color_frame,
    cho::core::PointCloud<float, 3>* const cloud_out,
    cho::core::PointCloud<std::uint8_t, 3>* const colors = nullptr) {
  // Hopefully nothing funny is going on with this.
  static_assert((sizeof(rs2::vertex) % sizeof(float)) == 0);

  const rs2::vertex* const vertices = cloud_in.get_vertices();
  const rs2::texture_coordinate* const tex_coords =
      cloud_in.get_texture_coordinates();

#if 0
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
#endif

#if 0
  // Allocate data.
  cloud_out->SetNumPoints(cloud_in.size());

  // Copy data.
  cloud_out->GetData() =
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>,
                 Eigen::Unaligned,
                 Eigen::Stride<1, sizeof(rs2::vertex) / sizeof(float)>>(
          const_cast<float*>(reinterpret_cast<const float*>(vertices)),
          cloud_in.size(), 3)
          .transpose();
#else
  cloud_out->SetNumPoints(cloud_in.size());
  colors->SetNumPoints(cloud_in.size());

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
    const int x =
        std::min(std::max(static_cast<int>(tex_coords[i].u * w), 0), w - 1);
    const int y =
        std::min(std::max(static_cast<int>(tex_coords[i].v * h), 0), h - 1);
    const int offset = x * color_frame.get_bytes_per_pixel() + y * stride;
    if (0 <= offset && offset + 2 < color_frame.get_data_size()) {
      colors->GetPoint(i).x() = color_ptr[offset];
      colors->GetPoint(i).y() = color_ptr[offset + 1];
      colors->GetPoint(i).z() = color_ptr[offset + 2];
    }
  }
#endif
}

class RsDriver::Impl {
 public:
  explicit Impl();
  virtual ~Impl();
  void Setup();
  void SetFrameRate(const float frame_rate);
  bool GetFrame(cv::Mat* const depth_image, cv::Mat* const color_image,
                cho::core::PointCloud<float, 3>* const point_cloud,
                cho::core::PointCloud<std::uint8_t, 3>* const color_cloud,
                std::int64_t* const timestamp);
  Eigen::Matrix3f GetIntrinsicMatrix();

 private:
  rs2::pointcloud cloud;
  rs2::points points;
  rs2::context ctx;
  std::vector<rs2::pipeline> pipelines;
  double frame_rate_{0};

  // latest data.
  double last_timestamp_{0};
  cv::Mat depth_image_;
  cv::Mat color_image_;
  cho::core::PointCloud<float, 3> point_cloud_;
  cho::core::PointCloud<std::uint8_t, 3> color_cloud_;

  // derived cache
  double frame_interval_{0};
  std::thread reader_thread_;
  std::shared_mutex data_mutex_;
  bool running_{false};
};

RsDriver::RsDriver() : impl_{std::make_unique<Impl>()} {}
RsDriver::~RsDriver() = default;
void RsDriver::Setup() { return impl_->Setup(); }
void RsDriver::SetFrameRate(const float frame_rate) {
  return impl_->SetFrameRate(frame_rate);
}
bool RsDriver::GetFrame(
    cv::Mat* const depth_image, cv::Mat* const color_image,
    cho::core::PointCloud<float, 3>* const point_cloud,
    cho::core::PointCloud<std::uint8_t, 3>* const color_cloud,
    std::int64_t* const timestamp) {
  return impl_->GetFrame(depth_image, color_image, point_cloud, color_cloud,
                         timestamp);
}
Eigen::Matrix3f RsDriver::GetIntrinsicMatrix() {
  return impl_->GetIntrinsicMatrix();
}

RsDriver::Impl::Impl() {}

RsDriver::Impl::~Impl() {
  running_ = false;
  if (reader_thread_.joinable()) {
    reader_thread_.join();
  }
}

void RsDriver::Impl::Setup() {
  fmt::print("Begin querying devices\n");
  for (const auto& dev : ctx.query_devices()) {
    rs2::pipeline pipe(ctx);
    rs2::config cfg;
    cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    pipe.start(cfg);
    pipelines.emplace_back(std::move(pipe));

    // FIXME(ycho-or): Currently only admitting a single device.
    break;
  }

  if (pipelines.empty()) {
    throw std::runtime_error(
        "No valid devices found! Failed to initialize pipelines.\n");
  }

  // Begin thread for reading data.
  running_ = true;
  reader_thread_ = std::thread([this]() {
    while (running_) {
      for (const auto& pipe : pipelines) {
        // Get input data.
        auto frames = pipe.wait_for_frames();
        auto color_frame = frames.get_color_frame();
        auto depth_frame = frames.get_depth_frame();
        if (depth_frame.get_timestamp() <= last_timestamp_ + frame_interval_) {
          continue;
        }

        // Construct point cloud from color+depth_frame.
        cloud.map_to(color_frame);
        points = cloud.calculate(depth_frame);
        std::vector<std::uint8_t> colors;

        // Convert to non-RS2 format.
        cho::core::PointCloud<float, 3> point_cloud;
        cho::core::PointCloud<std::uint8_t, 3> color_cloud;
        cv::Mat color_image;
        cv::Mat depth_image;
        ConvertPointCloud(points, color_frame, &point_cloud, &color_cloud);
        ConvertImage(color_frame, CV_8UC3, &color_image);
        ConvertImage(depth_frame, CV_16UC1, &depth_image);

        // Set data to internal struct.
        {
          std::unique_lock lock(data_mutex_);
          last_timestamp_ = depth_frame.get_timestamp();
          point_cloud_ = std::move(point_cloud);
          color_cloud_ = std::move(color_cloud);
          color_image_ = std::move(color_image);
          depth_image_ = std::move(depth_image);
        }
      }
    }
  });
}

void RsDriver::Impl::SetFrameRate(const float frame_rate) {
  frame_rate_ = frame_rate;
  frame_interval_ = (1.0 / frame_rate);
}

bool RsDriver::Impl::GetFrame(
    cv::Mat* const depth_image, cv::Mat* const color_image,
    cho::core::PointCloud<float, 3>* const point_cloud,
    cho::core::PointCloud<std::uint8_t, 3>* const color_cloud,
    std::int64_t* const timestamp) {
  std::shared_lock lock(data_mutex_);

  // No data, return
  if (last_timestamp_ == 0) {
    return false;
  }

  // Export frame ...
  if (point_cloud) {
    *point_cloud = point_cloud_;
  }
  if (color_cloud) {
    *color_cloud = color_cloud_;
  }
  return true;
}

Eigen::Matrix3f RsDriver::Impl::GetIntrinsicMatrix() {
  Eigen::Matrix3f out{Eigen::Matrix3f::Zero()};
  if (pipelines.empty()) {
    return out;
  }
  auto intrinsics = pipelines.front()
                        .get_active_profile()
                        .get_stream(RS2_STREAM_COLOR)
                        .as<rs2::video_stream_profile>()
                        .get_intrinsics();
  out.coeffRef(0, 0) = intrinsics.fx;
  out.coeffRef(0, 2) = intrinsics.ppx;
  out.coeffRef(1, 1) = intrinsics.fy;
  out.coeffRef(1, 2) = intrinsics.ppy;
  out.coeffRef(2, 2) = 1;
  return out;
}

}  // namespace rs_tracker
