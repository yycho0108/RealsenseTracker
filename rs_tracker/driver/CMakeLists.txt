# #############################################################################
# Simple realsense camera driver.
add_library(
  rs_driver SHARED
  src/rs_driver.cpp
  src/data_source_rs.cpp
  include/rs_tracker/driver/rs_driver.hpp
  include/rs_tracker/driver/data_source.hpp
  include/rs_tracker/driver/data_source_rs.hpp)

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
  ${OpenCV_LIBS}
  rs_common)
# #############################################################################
