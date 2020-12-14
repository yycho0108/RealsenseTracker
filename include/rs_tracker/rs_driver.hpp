#pragma once

#include <memory>

#include <Eigen/Geometry>
#include <cho_util/core/geometry/point_cloud.hpp>
#include <opencv2/opencv.hpp>

namespace rs_tracker {
class RsDriver {
 public:
  explicit RsDriver();
  virtual ~RsDriver();
  void Setup();

  void SetFrameRate(const float frame_rate);
  bool GetFrame(cv::Mat* const depth_image, cv::Mat* const color_image,
                cho::core::PointCloud<float, 3>* const point_cloud,
                cho::core::PointCloud<std::uint8_t, 3>* const color_cloud,
                std::int64_t* const timestamp);

  Eigen::Matrix3f GetIntrinsicMatrix();

 private:
  class Impl;
  friend class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace rs_tracker
