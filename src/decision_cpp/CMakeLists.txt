cmake_minimum_required(VERSION 3.8)
project(decision_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

add_executable(path_plan src/path_plan.cpp)
ament_target_dependencies(path_plan rclcpp std_msgs geometry_msgs nav_msgs)

add_executable(safety_supervisor src/safety_supervisor.cpp)
ament_target_dependencies(safety_supervisor rclcpp std_msgs geometry_msgs)

install(TARGETS
  path_plan
  safety_supervisor
  DESTINATION lib/${PROJECT_NAME})

ament_package()
