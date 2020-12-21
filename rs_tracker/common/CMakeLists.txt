# #############################################################################
# commonly used utils inside our app.
add_library(
  rs_common SHARED
  src/fpfh.cpp
  src/point_cloud_utils.cpp
  include/rs_tracker/common.hpp
  include/rs_tracker/common/fpfh.hpp
  include/rs_tracker/common/kdtree.hpp
  include/rs_tracker/common/point_cloud_utils.hpp
  include/rs_tracker/common/types.hpp)
target_include_directories(rs_common PUBLIC include)
target_include_directories(rs_common PRIVATE ${ChoUtil_INCLUDE_DIRS}
                                             ${Eigen_INCLUDE_DIRS})
target_link_libraries(rs_common fmt ChoUtil::core Eigen3::Eigen
                      Threads::Threads ${OpenCV_LIBS})
# #############################################################################
