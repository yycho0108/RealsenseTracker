#include "rs_tracker/rs_viewer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
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
#include <cho_util/vis/convert_proto.hpp>
#include <cho_util/vis/render_data.hpp>
#include <cho_util/vis/subprocess_viewer.hpp>
// #include <cho_util/vis/remote_viewer.hpp>

#include "rs_tracker/rs_driver.hpp"

// Declare impl class.
class RsViewer::Impl {
 public:
  explicit Impl(const RsViewerSettings& settings);
  virtual ~Impl();
  void SetupDriver();
  void SetupViewer();
  void Loop();

 private:
  RsViewerSettings settings_;
  cho::vis::SubprocessViewer viewer_;
  rs_tracker::RsDriver driver_;

  // cho::vis::RemoteViewerClient viewer;
  // cho::vis::RenderData render_data;
};

// Forward to impl.
RsViewer::RsViewer(const RsViewerSettings& settings)
    : impl_(std::make_unique<Impl>(settings)) {}
RsViewer::~RsViewer() {}
void RsViewer::SetupDriver() { return impl_->SetupDriver(); }
void RsViewer::SetupViewer() { return impl_->SetupViewer(); }
void RsViewer::Loop() { return impl_->Loop(); }

RsViewer::Impl::Impl(const RsViewerSettings& settings)
    : settings_(settings), viewer_{false} {}

RsViewer::Impl::~Impl() = default;

void RsViewer::Impl::SetupDriver() { driver_.Setup(); }

void RsViewer::Impl::SetupViewer() { viewer_.Start(); }

void RsViewer::Impl::Loop() {
  fmt::print("Begin processing loop\n");
  double last_timestamp{0};

  cv::Mat color_image;
  cv::Mat depth_image;
  cho::core::PointCloud<float, 3> point_cloud;
  cho::core::PointCloud<std::uint8_t, 3> color_cloud;
  std::int64_t timestamp{0};
  int count{0};

  while (true) {
    // Get frame.
    const bool have_frame = driver_.GetFrame(
        &depth_image, &color_image, &point_cloud, &color_cloud, &timestamp);
    if (!have_frame || timestamp <= last_timestamp) {
      continue;
    }

    // Render data.
    cho::vis::RenderData render_data{
        .tag = "cloud",
        .geometry = point_cloud,
        .color = {},  // hmm
        .opacity = 1.0f,
        .representation = cho::vis::RenderData::Representation::kPoints,
        .quit = false};
    viewer_.Render(render_data);

    // If enabled, save to file.
    if (!settings_.record_file.empty()) {
      auto cloud_pb =
          cho::type::Convert<cho::proto::core::geometry::PointCloud>(
              point_cloud);
      std::ofstream fout{fmt::format(settings_.record_file, count),
                         std::ios_base::out | std::ios_base::binary};
      cloud_pb.SerializeToOstream(&fout);
    }

    // Increment frame count.
    ++count;
  }
}
