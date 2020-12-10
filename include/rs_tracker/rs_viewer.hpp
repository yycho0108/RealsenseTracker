#pragma once

#include <memory>

struct RsViewerSettings {
  std::string record_file{""};
  float frame_interval_ms{1.0f};
};

struct RsViewer {
  explicit RsViewer(const RsViewerSettings& settings);
  virtual ~RsViewer();
  void SetupPipeline();
  void SetupViewer();
  void Loop();

 private:
  class Impl;
  friend class Impl;
  std::unique_ptr<Impl> impl_;
};
