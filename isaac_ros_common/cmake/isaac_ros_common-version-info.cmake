# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# isaac_ros_common/cmake/version_info.cmake

function(generate_version_info PROJECT_NAME)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Assume that this macro is being called from another package that has
    # a build dependency on isaac_ros_common.

    # Check if the project is 'isaac_ros_common' to use relative pathing
    if("${PROJECT_NAME}" STREQUAL "isaac_ros_common")
        # Use relative pathing
        set(SCRIPT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/scripts/isaac_ros_version_embed.py")
    else()
        # Use the package path resolution
        find_package(ament_cmake REQUIRED)
        ament_index_get_resource(ISAAC_ROS_COMMON_SCRIPTS_PATH isaac_ros_common_scripts_path isaac_ros_common)
        set(SCRIPT_PATH "${ISAAC_ROS_COMMON_SCRIPTS_PATH}/isaac_ros_version_embed.py")
    endif()

    # Output path for the version_info.yaml file
    set(OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/version_info.yaml")

    # Install destination for the generated YAML file
    set(INSTALL_DESTINATION "share/${PROJECT_NAME}")

    # Add a custom command to generate the version info YAML file
    add_custom_command(
        OUTPUT ${OUTPUT_PATH}
        COMMAND ${Python3_EXECUTABLE} ${SCRIPT_PATH} --output ${OUTPUT_PATH} --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS ${SCRIPT_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/package.xml
        COMMENT "Generating version information as YAML"
    )

    # Add a custom target that depends on the output file
    add_custom_target(
        generate_version_info_target_${PROJECT_NAME} ALL
        DEPENDS ${OUTPUT_PATH}
    )

    # Install the generated YAML file to the install directory
    install(FILES ${OUTPUT_PATH} DESTINATION ${INSTALL_DESTINATION})

endfunction()
