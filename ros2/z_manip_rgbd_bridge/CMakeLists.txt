cmake_minimum_required(VERSION 3.16)
project(z_manip_rgbd_bridge)

find_package(ament_cmake REQUIRED)
find_package(image_transport REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(nonlazy_rgbd_bridge src/nonlazy_rgbd_bridge.cpp)
target_compile_features(nonlazy_rgbd_bridge PUBLIC cxx_std_17)
target_compile_options(nonlazy_rgbd_bridge PRIVATE -Wall -Wextra -Wpedantic)
ament_target_dependencies(nonlazy_rgbd_bridge image_transport rclcpp sensor_msgs)

install(TARGETS nonlazy_rgbd_bridge DESTINATION lib/${PROJECT_NAME})

ament_package()
