# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# Create a target that installs a single asset file.
#  * An install script asset_scripts/{TARGET_NAME}.sh must be available.
#  * The script must support a "--print-install-paths" arg that does nothing except printing a
#    space-separated list of all outputs produced by the script. This is used to handle build
#    dependencies.
function(install_isaac_ros_asset TARGET_NAME)

  set(INSTALL_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/asset_scripts/${TARGET_NAME}.sh")

  if(NOT EXISTS "${INSTALL_SCRIPT}")
    message(FATAL_ERROR "File not found: ${INSTALL_SCRIPT}")
  endif()

  message(STATUS "Install script path: ${INSTALL_SCRIPT}")

  # Dry-run the install script to get the expected output path
  execute_process(COMMAND
    "${INSTALL_SCRIPT}" --print-install-paths
    RESULT_VARIABLE DRYRUN_RESULT
    OUTPUT_VARIABLE DRYRUN_STDOUT
    ERROR_VARIABLE DRYRUN_STDERR
  )

  if(NOT ${DRYRUN_RESULT} STREQUAL "0")
    message(FATAL_ERROR "${DRYRUN_STDOUT}\n${DRYRUN_STDERR}")
  endif()

  # Make a CMAKE list out of the output paths
  string(REPLACE " " ";" OUTPUT_PATHS ${DRYRUN_STDOUT})

  message(STATUS "Installing asset: ${OUTPUT_PATHS}")

  # Add a command with a cmake-recognized output that calls the script
  add_custom_command(
    OUTPUT ${OUTPUT_PATHS}
    COMMAND ${INSTALL_SCRIPT}
  )

  # Hook the command up with a target
  add_custom_target("${TARGET_NAME}" ALL DEPENDS ${OUTPUT_PATHS})
endfunction()




