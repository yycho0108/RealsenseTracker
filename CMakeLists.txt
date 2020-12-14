cmake_minimum_required(VERSION 3.14)
project(template)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# #############################################################################
# find relevant packages ...
find_package(Threads)
find_package(nanoflann REQUIRED)
find_package(realsense2 REQUIRED)
find_package(Ceres REQUIRED)
# find_package(PCL REQUIRED COMPONENTS COMMON IO OCTREE SEARCH FEATURES FILTERS
# SURFACE)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system program_options)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ChoUtil REQUIRED)
# #############################################################################

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
target_link_libraries(viewer fmt ChoUtil::core ChoUtil::vis Eigen3::Eigen)
# #############################################################################

# #############################################################################
# Simple realsense camera driver.
add_library(rs_driver SHARED src/rs_driver.cpp
                             include/rs_tracker/rs_driver.hpp)
target_include_directories(rs_driver PUBLIC include)
target_include_directories(
  rs_driver PRIVATE ${realsense2_INCLUDE_DIR} ${ChoUtil_INCLUDE_DIRS}
                    ${Eigen_INCLUDE_DIRS})
target_link_libraries(
  rs_driver
  fmt
  ${realsense2_LIBRARY}
  ChoUtil::core
  Eigen3::Eigen
  Threads::Threads
  ${OpenCV_LIBS})
# #############################################################################

# #############################################################################
# Actual viewer - lib
add_library(rs_viewer SHARED src/rs_viewer.cpp
                             include/rs_tracker/rs_viewer.hpp)
target_include_directories(rs_viewer PUBLIC include)
target_include_directories(rs_viewer PRIVATE ${ChoUtil_INCLUDE_DIRS}
                                             ${Eigen_INCLUDE_DIRS})
target_link_libraries(rs_viewer fmt ChoUtil::core ChoUtil::vis Eigen3::Eigen
                      rs_driver)
# #############################################################################

# #############################################################################
# Actual viewer - app
add_executable(rs_viewer_app src/rs_viewer_app.cpp)
target_include_directories(rs_viewer_app PUBLIC include)
target_include_directories(rs_viewer_app PRIVATE ${BOOST_INCLUDE_DIRS})
target_link_libraries(rs_viewer_app rs_viewer Boost::program_options)
# #############################################################################

# #############################################################################
# Replay app
add_executable(rs_replay_app src/rs_replay_app.cpp src/align_icp.cpp
                             src/point_cloud_utils.cpp)
target_include_directories(rs_replay_app PUBLIC include)
target_include_directories(rs_replay_app PRIVATE ${BOOST_INCLUDE_DIRS})
target_link_libraries(
  rs_replay_app PRIVATE fmt ChoUtil::core ChoUtil::vis Eigen3::Eigen
                        Boost::program_options)
# #############################################################################

# #############################################################################
# Tracker app
add_executable(
  rs_tracker
  src/rs_tracker.cpp
  src/align_gicp.cpp
  src/point_cloud_utils.cpp
  src/data_source_rs.cpp
  include/rs_tracker/align_gicp.hpp
  include/rs_tracker/common.hpp
  include/rs_tracker/gicp_cost.hpp
  include/rs_tracker/point_cloud_utils.hpp
  include/rs_tracker/data_source.hpp
  include/rs_tracker/data_source_rs.hpp)
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
  Eigen3::Eigen)
# #############################################################################

# #############################################################################
# Compile commands ###
add_custom_target(generate-compile-commands
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/compile_commands_compdb.json
  COMMAND compdb -p ${CMAKE_CURRENT_BINARY_DIR} list >
          ${CMAKE_CURRENT_BINARY_DIR}/compile_commands_compdb.json
  DEPENDS generate-compile-commands
          ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
  COMMENT "Export compile commands also for headers"
  VERBATIM)
add_custom_target(
  generate-compdb-compile-commands ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/compile_commands_compdb.json)
# #############################################################################

# #############################################################################
# Copy scripts.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
  add_custom_target(
    symlink_scripts ALL
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/"
      "${CMAKE_CURRENT_BINARY_DIR}/scripts")
endif()
# #############################################################################

# #############################################################################
# Debugging - print all variables.
if(FALSE)
  get_cmake_property(_variableNames VARIABLES)
  list(SORT _variableNames)
  foreach(_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endif()
# #############################################################################
