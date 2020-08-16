cmake_minimum_required(VERSION 3.14)
project(template)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_package(nanoflann REQUIRED)
find_package(realsense2 REQUIRED)
find_package(Ceres REQUIRED)

# find_package(PCL REQUIRED COMPONENTS COMMON IO OCTREE SEARCH FEATURES FILTERS
# SURFACE)

find_package(Eigen3 REQUIRED)

# find_package(Boost REQUIRED COMPONENTS system)

find_package(fmt REQUIRED)

# find_package(OpenCV REQUIRED)

find_package(ChoUtil REQUIRED)

add_executable(basic_capture src/basic_capture.cpp)
target_include_directories(basic_capture PUBLIC include)
target_include_directories(basic_capture PRIVATE ${realsense2_INCLUDE_DIR})
target_link_libraries(basic_capture fmt ${realsense2_LIBRARY})

add_executable(viewer src/view_xyzrgb.cpp)
target_include_directories(viewer PRIVATE ${ChoUtil_INCLUDE_DIRS})
target_link_libraries(viewer fmt ChoUtil::core ChoUtil::vis Eigen3::Eigen)

add_executable(rs_viewer src/rs_viewer.cpp)
target_include_directories(rs_viewer PUBLIC include)
target_include_directories(
  rs_viewer PRIVATE ${realsense2_INCLUDE_DIR} ${ChoUtil_INCLUDE_DIRS}
                    ${Eigen_INCLUDE_DIRS})
target_link_libraries(rs_viewer fmt ${realsense2_LIBRARY} ChoUtil::core
                      ChoUtil::vis Eigen3::Eigen)

add_executable(
  rs_tracker
  src/rs_tracker.cpp
  src/align.cpp
  src/point_cloud_utils.cpp
  src/data_source_rs.cpp
  include/rs_tracker/align.hpp
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

# Copy scripts.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
  add_custom_target(
    symlink_scripts ALL
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink
      "${CMAKE_CURRENT_SOURCE_DIR}/scripts/"
      "${CMAKE_CURRENT_BINARY_DIR}/scripts")
endif()

# Debugging - print all variables.
if(TRUE)
  get_cmake_property(_variableNames VARIABLES)
  list(SORT _variableNames)
  foreach(_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endif()
