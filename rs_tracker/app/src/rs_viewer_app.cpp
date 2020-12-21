#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>

#include <fmt/printf.h>

#include "rs_tracker/vis/rs_viewer.hpp"

namespace po = boost::program_options;

void Usage(const po::options_description& desc) {
  // clang-format off
  fmt::print(R"(Usage:
  rs_viewer_app [options]
Description:
  Viewer app for realsense.
Options:
{}
  )",
             desc);
  // clang-format on
}

bool ParseArguments(int argc, char* argv[], RsViewerSettings* const settings) {
  po::options_description desc("");

  // clang-format off
  desc.add_options()
      ("help,h", "help")
      ("record,r", po::value<std::string>(&settings->record_file)->default_value(""), "Output record file pattern, i.e. /tmp/{:04d}.pb")
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

int main(int argc, char* argv[]) {
  RsViewerSettings settings;
  if (!ParseArguments(argc, argv, &settings)) {
    return 1;
  }

  RsViewer viewer{settings};
  viewer.SetupDriver();
  viewer.SetupViewer();
  viewer.Loop();
  return 0;
}
