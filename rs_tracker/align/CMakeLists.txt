# #############################################################################

# Interface lib
add_library(
  teaser_interface SHARED include/rs_tracker/align/teaser_interface.hpp
                          src/teaser_interface.cpp)
target_include_directories(
  teaser_interface
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(teaser_interface PRIVATE teaser Eigen3::Eigen)
target_compile_options(teaser_interface PRIVATE ${TEASER_COMPILE_OPTIONS})

# Simple realsense camera driver.
add_library(rs_align SHARED src/align_icp.cpp src/align_gicp.cpp)
target_include_directories(rs_align PUBLIC include)
target_include_directories(
  rs_align PRIVATE ${realsense2_INCLUDE_DIR} ${ChoUtil_INCLUDE_DIRS}
                   ${Eigen_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
target_link_libraries(
  rs_align
  PRIVATE fmt
          ${realsense2_LIBRARY}
          ChoUtil::core
          Eigen3::Eigen
          Threads::Threads
          ${OpenCV_LIBS}
          ${CERES_LIBRARIES}
          rs_common)
target_link_libraries(rs_align PUBLIC teaser_interface)
# #############################################################################
