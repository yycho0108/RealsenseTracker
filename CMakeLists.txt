# #############################################################################
# Configure project.
cmake_minimum_required(VERSION 3.14)
project(template)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
# #############################################################################

# #############################################################################
# Find packages.
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

add_subdirectory(rs_tracker)
add_subdirectory(third_party)

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
