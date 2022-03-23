# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.5)
project(isaac_ros_nvengine LANGUAGES C CXX)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )

execute_process(COMMAND uname -m COMMAND tr -d '\n'
  OUTPUT_VARIABLE ARCHITECTURE
)
message( STATUS "Architecture: ${ARCHITECTURE}" )

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# GXF
include_directories(gxf/include)
add_library(gxf SHARED IMPORTED)
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  set_property(TARGET gxf PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/gxf/lib/gxf_x86_64_cuda_11_4/core/libgxf_core.so)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  set_property(TARGET gxf PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/gxf/lib/gxf_jetpack46_1/core/libgxf_core.so)
endif()

# gxe_node
ament_auto_add_library(gxe_node SHARED src/gxe_node.cpp)
target_compile_definitions(gxe_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(gxe_node gxf)
rclcpp_components_register_nodes(gxe_node "isaac_ros::nvengine::GXENode")
set(node_plugins "${node_plugins}isaac_ros::nvengine::GXENode;$<TARGET_FILE:gxe_node>\n")

# ROS tensor publisher node
ament_auto_add_library(ros_tensor_publisher_node SHARED src/ros_tensor_publisher_node.cpp)
target_compile_definitions(ros_tensor_publisher_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
target_link_libraries(ros_tensor_publisher_node gxf)
rclcpp_components_register_nodes(ros_tensor_publisher_node "isaac_ros::nvengine::ROSTensorPublisherNode")
set(node_plugins "${node_plugins}isaac_ros::nvengine::ROSTensorPublisherNode;$<TARGET_FILE:ros_tensor_publisher_node>\n")

# test ping executable
ament_auto_add_executable("test_tensor"
  src/nvengine_main.cpp
)

target_link_libraries("test_tensor" gxe_node)

# test publish executable
ament_auto_add_executable("test_publish_tensor"
  src/tensor_publish.cpp
)

target_link_libraries("test_publish_tensor" ros_tensor_publisher_node)

# Install config directory
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install extensions directory
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  install(DIRECTORY gxf/lib/gxf_x86_64_cuda_11_4/ DESTINATION share/${PROJECT_NAME}/gxf)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  install(DIRECTORY gxf/lib/gxf_jetpack46_1/ DESTINATION share/${PROJECT_NAME}/gxf)
endif()
set_target_properties(gxe_node PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(gxe_node PROPERTIES INSTALL_RPATH "$ORIGIN/../share/${PROJECT_NAME}/gxf/core")
set_target_properties(ros_tensor_publisher_node PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(ros_tensor_publisher_node PROPERTIES INSTALL_RPATH "$ORIGIN/../share/${PROJECT_NAME}/gxf/core")

# Install package executable
install(TARGETS "test_tensor"
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS "test_publish_tensor"
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Ignore copyright notices since we use custom NVIDIA Isaac ROS Software License
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_nvengine_pol.py)
endif()

ament_auto_package()
