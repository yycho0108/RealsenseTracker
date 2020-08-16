#include <fmt/printf.h>
#include <fstream>

#include <cho_util/proto/render.pb.h>
#include <cho_util/core/geometry.hpp>
#include <cho_util/core/geometry/line.hpp>
#include <cho_util/core/geometry/point_cloud.hpp>
#include <cho_util/core/geometry/sphere.hpp>
#include <cho_util/type/convert.hpp>
// #include <cho_util/vis/direct_viewer.hpp>
#include <cho_util/vis/render_data.hpp>
#include <cho_util/vis/subprocess_viewer.hpp>

void LoadXyzrgb(const std::string& filename,
                cho::core::PointCloud<float, 3>* const cloud,
                std::vector<std::uint8_t>* const color) {
  std::ifstream fin(filename);
  float x, y, z, r, g, b;
  std::vector<float> data;
  color->clear();
  while (fin) {
    fin >> x >> y >> z >> r >> g >> b;
    data.emplace_back(x);
    data.emplace_back(y);
    data.emplace_back(z);

    color->emplace_back(static_cast<std::uint8_t>(r * 255));
    color->emplace_back(static_cast<std::uint8_t>(g * 255));
    color->emplace_back(static_cast<std::uint8_t>(b * 255));
  }
  if (data.empty()) {
    throw std::invalid_argument("file does not contain valid data!");
  }
  cloud->SetNumPoints(data.size() / 3);
  cloud->GetData() =
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>(
          data.data(), data.size() / 3, 3)
          .transpose();
}

int main() {
  cho::core::PointCloud<float, 3> cloud_geom;
  cho::vis::RenderData cloud{
      .tag = "cloud",
      .geometry = cloud_geom,
      .color = {255, 255, 255},
      .representation = cho::vis::RenderData::Representation::kPoints,
      .quit = false};
  cho::vis::SubprocessViewer viewer{true};

  std::vector<std::uint8_t> color;
  for (int i = 0; i < 100; ++i) {
    const std::string filename = fmt::format("/tmp/{:04d}.xyzrgb", i + 1);
    fmt::print("Loading from = {}\n", filename);
    LoadXyzrgb(filename, &cloud_geom, &color);
    cloud.geometry = cloud_geom;
    // cloud.color = color;
    viewer.Render(cloud);
  }

  return EXIT_SUCCESS;
}
