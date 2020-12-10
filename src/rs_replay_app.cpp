
#include <chrono>
#include <fstream>
#include <ios>
#include <iostream>
#include <thread>

#include <cho_util/proto/geometry.pb.h>
#include <fmt/printf.h>
#include <glob.h>

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>

#include <cho_util/proto/render.pb.h>
#include <cho_util/core/geometry/point_cloud.hpp>

#include <cho_util/type/convert.hpp>
#include <cho_util/vis/convert_proto.hpp>
#include <cho_util/vis/render_data.hpp>
#include <cho_util/vis/subprocess_viewer.hpp>

#include "rs_tracker/rs_viewer.hpp"

namespace po = boost::program_options;

struct RsReplayAppSettings {
  std::string record_file;
  float frame_interval_ms;
};

void Usage(const po::options_description& desc) {
  // clang-format off
  fmt::print(R"(Usage:
  rs_replay_app [options]
Description:
  Replay app for realsense record.
Options:
{}
  )",
             desc);
  // clang-format on
}

bool ParseArguments(int argc, char* argv[],
                    RsReplayAppSettings* const settings) {
  po::options_description desc("");

  // clang-format off
  desc.add_options()
      ("help,h", "help")
      ("record,r", po::value<std::string>(&settings->record_file)->default_value(""), "Input record file pattern, i.e. /tmp/{:04d}.pb")
      ("frame_interval,f", po::value<float>(&settings->frame_interval_ms)->default_value(1000.0f), "Frame interval, in milliseconds.")
      ;
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    Usage(desc);
    return false;
  }
  return true;
}

std::vector<std::string> Glob(const std::string& pattern) {
  using namespace std;

  // glob struct resides on the stack
  glob_t glob_result;
  memset(&glob_result, 0, sizeof(glob_result));

  // do the glob operation
  int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  if (return_value != 0) {
    globfree(&glob_result);
    stringstream ss;
    ss << "glob() failed with return_value " << return_value << endl;
    throw std::runtime_error(ss.str());
  }

  // collect all the filenames into a std::list<std::string>
  vector<string> filenames;
  for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
    filenames.push_back(string(glob_result.gl_pathv[i]));
  }

  // cleanup
  globfree(&glob_result);

  // done
  return filenames;
}

int main(int argc, char* argv[]) {
  RsReplayAppSettings settings;
  if (!ParseArguments(argc, argv, &settings)) {
    return 1;
  }
  std::vector<std::string> record_files = Glob(settings.record_file);
  std::sort(record_files.begin(), record_files.end());

  // instantiate viewer
  cho::vis::SubprocessViewer viewer{false};
  viewer.Start();
  int index{0};
  while (1) {
    // Sanitize index against wrap-around
    if (index >= record_files.size()) {
      index %= record_files.size();
    }

    // Read data.
    const std::string record_file = record_files.at(index);
    std::ifstream fin{record_file, std::ios_base::binary | std::ios_base::in};
    cho::proto::core::geometry::PointCloud cloud_proto;
    cloud_proto.ParseFromIstream(&fin);
    const auto& cloud =
        cho::type::Convert<cho::core::PointCloud<float, 3>>(cloud_proto);

    // Render.
    cho::vis::RenderData render_data{
        .tag = "cloud",
        .geometry = cloud,
        .color = {},
        .representation = cho::vis::RenderData::Representation::kPoints,
        .quit = false};
    viewer.Render(render_data);

    // Increment index.
    ++index;

    // Sleep.
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int>(settings.frame_interval_ms)));
  }

  return 0;
}
