#pragma once

#include <cho_util/core/geometry/point_cloud.hpp>

#include "rs_tracker/common.hpp"

namespace rs_tracker {

template <typename Derived>
class DataSource {
 public:
  explicit DataSource() = default;
  ~DataSource() = default;
  friend Derived;
  bool GetCloud(const double prev_stamp,
                cho::core::PointCloud<float, 3>* const cloud,
                double* const curr_stamp) {
    return static_cast<Derived*>(this)->GetCloud(prev_stamp, cloud);
  }
};

class RandomSource : DataSource<RandomSource> {
 public:
  explicit RandomSource(){};
  explicit RandomSource(const int size, const double& timestep)
      : size_(size), timestep_(timestep) {}
  void SetSize(const int size) { size_ = size; }
  void SetStep(const double step) { timestep_ = step; }
  bool GetCloud(const double prev_stamp,
                cho::core::PointCloud<float, 3>* const cloud,
                double* const curr_stamp) {
    cloud->SetNumPoints(size_);
    cloud->GetData().setRandom();
    *curr_stamp = prev_stamp + timestep_;
    return true;
  }

 private:
  int size_;
  double timestep_;
};

}  // namespace rs_tracker
