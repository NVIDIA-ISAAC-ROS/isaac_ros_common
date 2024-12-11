# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Parse the arguments for add_graph_startup_test()
macro(parse_add_graph_startup_test_arguments namespace)
    cmake_parse_arguments(${namespace}
    "DONT_CHECK_EXIT_CODE"
    "TIMEOUT;ISAAC_ROS_GRAPH_STARTUP_TEST_PATH" # One value keywords
    "ARGS" # Multi value keywords
    ${ARGN})

    if(NOT ${namespace}_TIMEOUT)
        # The default timeout is 5 seconds.
        set(${namespace}_TIMEOUT 5)
    endif()

endmacro()

# Add a startup test for a launchfile.
#
# :param TARGET_NAME: The tests target name.
# :type TARGET_NAME: string
# :param LAUNCHFILE: The launchfile to test.
# :type LAUNCHFILE: string
# :param TIMEOUT: The test timeout in seconds.
# :type TIMEOUT: integer
# :param ISAAC_ROS_GRAPH_STARTUP_TEST_PATH: The path to the graph startup test, if manual
#    specification is desired. If not specified, which is the usual case, this
#    variable is determined automatically.
# :type ISAAC_ROS_GRAPH_STARTUP_TEST_PATH: string
# :param ARGS: Launch arguments to be passed to graph under test.
# :type ARGS: string
function(add_graph_startup_test TARGET_NAME LAUNCHFILE)
    parse_add_graph_startup_test_arguments(_add_graph_startup_test ${ARGN})

    # Path to the underlying meta-test.
    if(DEFINED _add_graph_startup_test_ISAAC_ROS_GRAPH_STARTUP_TEST_PATH)
        set(ISAAC_ROS_GRAPH_STARTUP_TEST_PATH ${_add_graph_startup_test_ISAAC_ROS_GRAPH_STARTUP_TEST_PATH})
    else()
        ament_index_has_resource(HAS_GRAPH_STARTUP_TEST_PATH graph_startup_test isaac_ros_test_cmake)
        if(NOT HAS_GRAPH_STARTUP_TEST_PATH)
            message(FATAL_ERROR "graph_startup_test resource not found.")
        endif()
        ament_index_get_resource(ISAAC_ROS_GRAPH_STARTUP_TEST_PATH graph_startup_test isaac_ros_test_cmake)
        if(EXISTS ${ISAAC_ROS_GRAPH_STARTUP_TEST_PATH})
            message(STATUS "Found the launchfile independent test at: ${ISAAC_ROS_GRAPH_STARTUP_TEST_PATH}")
        else()
            message(FATAL_ERROR "Could not find the launchfile independent test at: ${ISAAC_ROS_GRAPH_STARTUP_TEST_PATH}")
        endif()
    endif()


    # The package under test is the project name of the caller.
    set(PACKAGE ${PROJECT_NAME})

    # Compose arguments to the underlying test
    # Note that the arguments which are intended for the graph under test are grouped as a single string
    # and passed through the test as a single ROS argument "launch_file_arguments".
    set(ARGUMENTS "package_under_test:=${PACKAGE}" "launch_file_under_test:=${LAUNCHFILE}")
    list(APPEND ARGUMENTS "timeout:='${_add_graph_startup_test_TIMEOUT}'")
    if(${_add_graph_startup_test_DONT_CHECK_EXIT_CODE})
        list(APPEND ARGUMENTS "check_exit_code:='False'")
    endif()
    list(APPEND ARGUMENTS "launch_file_arguments:='${_add_graph_startup_test_ARGS}'")

    # Inside the test we trigger the graph to shutdown after the requested timeout.
    # However, launch_testing has a timeout at which point the test is killed.
    # We (arbitrarily) set this as 10 seconds longer than the requested timeout.
    math(EXPR LAUNCH_TESTING_TIMEOUT "${_add_graph_startup_test_TIMEOUT} + 20")

    add_launch_test(
        ${ISAAC_ROS_GRAPH_STARTUP_TEST_PATH}
        TARGET ${TARGET_NAME}
        ARGS ${ARGUMENTS}
        TIMEOUT ${LAUNCH_TESTING_TIMEOUT}
    )
endfunction()


# Return the installed path of a dummy bag that comes with this package.
#
# :param VAR: The the output variable that will hold the path.
# :type VAR: string
function(get_dummy_bag_path VAR)
  ament_index_get_resource(DUMMY_BAG_PATH dummy_bag isaac_ros_test_cmake)
  set(${VAR} ${DUMMY_BAG_PATH} PARENT_SCOPE)
endfunction()
