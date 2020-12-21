# #############################################################################
# Actual viewer - lib
add_library(rs_vis SHARED src/rs_viewer.cpp
                          include/rs_tracker/vis/rs_viewer.hpp)
target_include_directories(rs_vis PUBLIC include)
target_include_directories(rs_vis PRIVATE ${ChoUtil_INCLUDE_DIRS}
                                          ${Eigen_INCLUDE_DIRS})
target_link_libraries(
  rs_vis
  fmt
  ChoUtil::core
  ChoUtil::vis
  Eigen3::Eigen
  rs_driver
  rs_common)
# #############################################################################
