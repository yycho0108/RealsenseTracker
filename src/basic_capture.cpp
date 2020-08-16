#include <algorithm>
#include <iostream>
#include <librealsense2/rs.hpp>

#include <fmt/format.h>
#include <fmt/printf.h>

int main() {
  rs2::pointcloud cloud;
  rs2::points points;
  rs2::pose_frame pose_frame(nullptr);
  rs2::context ctx;
  std::vector<rs2::pipeline> pipelines;

  fmt::print("Begin querying devices\n");
  for (const auto& dev : ctx.query_devices()) {
    rs2::pipeline pipe(ctx);
    rs2::config cfg;
    cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    pipe.start(cfg);
    pipelines.emplace_back(std::move(pipe));
  }

  if (pipelines.empty()) {
    fmt::print(std::cerr,
               "No valid devices found! Failed to initialize pipelines.\n");
    return EXIT_FAILURE;
  }

  fmt::print("Begin processing loop\n");
  int count = 0;
  while (true) {
    ++count;
    fmt::print("\rcount={}", count);
    for (const auto& pipe : pipelines) {
      // Get input.
      auto frames = pipe.wait_for_frames();
      auto color = frames.get_color_frame();
      auto depth = frames.get_depth_frame();

      // Construct point cloud from color+depth.
      cloud.map_to(color);
      points = cloud.calculate(depth);

      points.export_to_ply(fmt::format("/tmp/{:04d}.ply", count), color);
      // fmt::print("cloud size = {}\n", points.size());
    }

    if (count >= 100) {
      break;
    }
  }
}
