cmake_minimum_required(VERSION 3.8)
project(control_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

add_executable(path_track src/path_track.cpp)
ament_target_dependencies(path_track rclcpp std_msgs geometry_msgs nav_msgs)

add_executable(cmd_mux src/cmd_mux.cpp)
ament_target_dependencies(cmd_mux rclcpp std_msgs geometry_msgs)

install(TARGETS
  path_track
  cmd_mux
  DESTINATION lib/${PROJECT_NAME})

ament_export_dependencies(
  geometry_msgs
  nav_msgs
  rclcpp
  std_msgs
)

ament_package()
