#pragma once

#include <librealsense2/rs.hpp>

#include "rs_tracker/driver/data_source.hpp"

namespace rs_tracker {

class RealsenseSource : public DataSource<RealsenseSource> {
 public:
  explicit RealsenseSource(const double min_interval_ms = 1000.0f);
  bool GetCloud(const double prev_timestamp,
                cho::core::PointCloud<float, 3>* const cloud,
                double* const curr_timestamp);

 private:
  // Settings.
  double min_interval_ms_;

  // Processing/device handles.
  rs2::pointcloud proc_cloud;
  rs2::points points;
  rs2::context ctx;
  std::vector<rs2::pipeline> pipelines;
};

}  // namespace rs_tracker
