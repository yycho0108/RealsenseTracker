# #############################################################################
# Simple demo app for storing point clouds.
add_executable(basic_capture src/basic_capture.cpp)
target_include_directories(basic_capture PUBLIC include)
target_include_directories(basic_capture PRIVATE ${realsense2_INCLUDE_DIR})
target_link_libraries(basic_capture fmt ${realsense2_LIBRARY})
# #############################################################################

# #############################################################################
# Simple viewer app (prototype?)
add_executable(viewer src/view_xyzrgb.cpp)
target_include_directories(viewer PRIVATE ${ChoUtil_INCLUDE_DIRS})
target_link_libraries(viewer fmt ChoUtil::core ChoUtil::vis Eigen3::Eigen
                      rs_vis)
# #############################################################################

# #############################################################################
# Actual viewer app
add_executable(rs_viewer_app src/rs_viewer_app.cpp)
target_include_directories(rs_viewer_app PUBLIC include)
target_include_directories(rs_viewer_app PRIVATE ${BOOST_INCLUDE_DIRS})
target_link_libraries(rs_viewer_app Boost::program_options rs_common rs_vis
                      rs_driver)
# #############################################################################

# #############################################################################
# Replay app
add_executable(rs_replay_app src/rs_replay_app.cpp)
target_include_directories(rs_replay_app PUBLIC include)
target_include_directories(rs_replay_app PRIVATE ${BOOST_INCLUDE_DIRS})
target_link_libraries(
  rs_replay_app
  PRIVATE fmt
          ChoUtil::core
          ChoUtil::vis
          Eigen3::Eigen
          Boost::program_options
          ${CERES_LIBRARIES}
          teaser_interface
          rs_common
          rs_align
          rs_vis)
# #############################################################################

# #############################################################################
# Alignment app
add_executable(rs_align_app src/rs_align_app.cpp)
target_include_directories(rs_align_app PUBLIC include)
target_include_directories(rs_align_app PRIVATE ${BOOST_INCLUDE_DIRS})
target_link_libraries(
  rs_align_app
  PRIVATE fmt
          ChoUtil::core
          ChoUtil::vis
          Eigen3::Eigen
          Boost::program_options
          ${CERES_LIBRARIES}
          teaser_interface
          rs_common
          rs_align
          rs_vis)
# #############################################################################

# #############################################################################
# Tracker app
add_executable(rs_tracker src/rs_tracker.cpp)
target_include_directories(rs_tracker PUBLIC include)
target_include_directories(
  rs_tracker
  PRIVATE ${CERES_INCLUDE_DIRS} ${realsense2_INCLUDE_DIR}
          ${ChoUtil_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS}
          ${nanoflann_INCLUDE_DIR})
target_link_libraries(
  rs_tracker
  fmt
  ${realsense2_LIBRARY}
  ${CERES_LIBRARIES}
  ChoUtil::core
  ChoUtil::vis
  Eigen3::Eigen
  rs_common
  rs_align
  rs_driver
  rs_vis)
# #############################################################################
