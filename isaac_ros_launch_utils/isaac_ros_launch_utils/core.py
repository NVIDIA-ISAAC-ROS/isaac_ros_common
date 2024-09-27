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
import argparse
from enum import Enum
import pathlib
import platform
import uuid
from typing import Any, Callable, List, Tuple, Dict
import yaml
import os

from ament_index_python.packages import get_package_share_directory

from isaac_ros_launch_utils.all_types import *


class NovaRobot(Enum):
    """Enum defining the type of nova robot."""

    NOVA_CARTER = 1
    NOVA_DEVELOPER_KIT = 2
    NOVA_BENCHTOP = 3
    UNKNOWN = 4


def _add_delay_if_set(action: Action, delay: Any = None) -> Action:
    if is_valid(delay):
        delay = float(delay) if isinstance(delay, str) else delay
        return TimerAction(period=delay, actions=[action])
    return action


class ArgumentContainer(argparse.Namespace):
    """
    A helper class to make it easier to define launch arguments and easier to see what arguments
    can be used with a launch graph.
    """

    def __init__(self):
        self._launch_configurations = []
        self._cli_launch_args = []
        self._opaque_functions = []

    def add_arg(self,
                name: str,
                default: Any = None,
                description: str | None = None,
                choices: list[str] | None = None,
                cli: bool = False) -> LaunchConfiguration:
        """ Add an argument to the arg container. """
        default = str(default) if default is not None else None
        launch_configuration = LaunchConfiguration(name, default=default)
        self._launch_configurations.append(launch_configuration)
        setattr(self, name, launch_configuration)
        if cli:
            self._cli_launch_args.append(
                DeclareLaunchArgument(
                    name,
                    default_value=default,
                    description=description,
                    choices=choices,
                ))
        return launch_configuration

    def get_launch_actions(self) -> list[Action]:
        """ Get all launch actions contained in this argument container. """
        return self._cli_launch_args + self._opaque_functions

    def add_opaque_function(
        self,
        function: Callable[['ArgumentContainer'], list[Action] | None],
    ) -> OpaqueFunction:
        """
        Helper function to add an opaque function that has access to all the evaluated arguments.
        """

        def helper_function(context: LaunchContext):
            evaluated_args = argparse.Namespace()
            for launch_configuration in self._launch_configurations:
                name = launch_configuration.variable_name[0].perform(context)
                value_str = launch_configuration.perform(context)
                try:
                    #pylint: disable=eval-used,
                    value = eval(value_str)
                #pylint: disable=bare-except,
                except:
                    value = value_str
                setattr(evaluated_args, name, value)
            return function(evaluated_args)

        opaque_function = OpaqueFunction(function=helper_function)
        self._opaque_functions.append(opaque_function)
        return opaque_function


def get_path(package: str, path: str) -> pathlib.Path:
    """ Get the path of an installed share file. """
    package_dir = pathlib.Path(get_package_share_directory(package))
    launch_path = package_dir / path
    return launch_path


def add_robot_description(
        nominals_package: Any,
        nominals_file: Any,
        robot_calibration_path: Any = "/etc/nova/calibration/isaac_calibration.urdf",
        override_path: Any = None) -> Action:
    """
    Loads an URDF file and adds a robot state publisher node.
    We select the first existing URDF file based on the following priorities:
    1) override urdf
    2) calibration urdf
    3) nominals urdf

    Args:
        nominals_package (Any): Package containing the nominals URDF file.
        nominals_file (Any): URDF nominals path, within nominals_package.
        robot_calibration_path (Any): Path to the URDF calibration file in the robot.
        override_path (Any): Path to the URDF override file.

    """

    def impl(context: LaunchContext) -> Action:
        nominals_package_str = perform_context(context, nominals_package)
        nominals_file_str = perform_context(context, nominals_file)
        robot_calibration_path_str = perform_context(context, robot_calibration_path)
        override_path_str = perform_context(context, override_path)

        override_urdf = pathlib.Path(override_path_str or '')
        calibrated_urdf = pathlib.Path(robot_calibration_path_str)
        nominals_urdf = get_path(nominals_package_str, nominals_file_str)

        if is_valid(override_path_str):
            if not override_urdf.is_file():
                raise FileNotFoundError(f'[add_robot_description]: Path of override URDF ' \
                        f'{override_path_str} does not exist.')
            print(f"Using override URDF from: {override_path_str}")
            urdf_path = override_urdf
        elif calibrated_urdf.is_file():
            print(f"Using calibrated URDF from: {robot_calibration_path}")
            urdf_path = calibrated_urdf
        elif nominals_urdf.is_file():
            print("Using nominals URDF")
            urdf_path = nominals_urdf
        else:
            raise Exception(f'No robot description found.')

        robot_description = Command(['xacro ', str(urdf_path)])

        robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': ParameterValue(robot_description, value_type=str),
            }],
        )
        return [robot_state_publisher]

    return OpaqueFunction(function=impl)


def include(package: str,
            path: str,
            launch_arguments: dict | None = None,
            condition: Condition = None,
            delay: float | None = None):
    """ Include another launch file. """
    launch_path = get_path(package, path)
    if path.endswith('.py'):
        source = PythonLaunchDescriptionSource([str(launch_path)])
    elif path.endswith('.xml'):
        source = XMLLaunchDescriptionSource([str(launch_path)])

    # Convert values to strings because launch arguments can only be strings.
    def make_valid_launch_argument(value: Any):
        if isinstance(value, (LaunchConfiguration, Substitution)):
            return value
        return str(value)

    launch_arguments = {
        k: make_valid_launch_argument(v) for k, v in (launch_arguments or {}).items()
    }

    include_action = IncludeLaunchDescription(
        source,
        launch_arguments=(launch_arguments or {}).items(),
        condition=condition,
    )

    action = _add_delay_if_set(include_action, delay)
    return action


def load_composable_nodes(container_name: str,
                          composable_nodes: list[ComposableNode],
                          log_message: Any = None,
                          condition: Condition = None) -> Action:
    """" Add a GroupAction that loads composable nodes and a log info depending on a condition. """
    actions = []
    actions.append(
        LoadComposableNodes(
            target_container=container_name,
            composable_node_descriptions=composable_nodes,
        ))

    if log_message is None:
        node_names = ['[ ']
        for node in composable_nodes:
            node_names.append('/')
            if node.node_namespace is not None:
                node_names.extend(node.node_namespace)
                node_names.append('/')
            if node.node_name is not None:
                node_names.extend(node.node_name)
            node_names.append(', ')
        node_names.pop()  # remove trailing comma and space
        node_names.append(' ]')
        log_message = ['Adding nodes '] + node_names + [' to container ', container_name, '.']

    actions.append(log_info(log_message))

    return GroupAction(actions, condition=condition)


def get_default_negotiation_time(x86_negotiation_time_s: int = 5,
                                 aarch64_negotiation_time_s: int = 20) -> int:
    """ Get a default negotiation time depending on the platform (x86/aarch64). """
    arch = platform.machine()
    negotiation_time = None
    if arch == 'x86_64':
        negotiation_time = x86_negotiation_time_s
    elif arch == 'aarch64':
        negotiation_time = aarch64_negotiation_time_s
    else:
        print(f'Warning platform type: {arch} not recognized. Using aarch64 negotiation time.')
        negotiation_time = aarch64_negotiation_time_s
    print(f'Platform detected as: {arch}. Setting negotiation time to: {negotiation_time}')
    return negotiation_time


def component_container(container_name: str,
                        log_level: str = 'info',
                        prefix=None,
                        condition=None,
                        container_type='multithreaded'):
    """" Add a component container. """
    # NOTE: We expose various component_container types, including the isolated component container
    # with the `use_multi_threaded_executor` argument. The reason is that we saw:
    # - issues with component_container_mt related to https://github.com/ros2/rclcpp/issues/2242.
    # - some GXF components don't function when using `component_container_isolated` *without*
    #   the `--use_multi_threaded_executor` flag.
    arguments = []
    if container_type == 'multithreaded':
        container_executable = 'component_container_mt'
    elif container_type == 'isolated':
        container_executable = 'component_container_isolated'
    elif container_type == 'isolated_multithreaded':
        container_executable = 'component_container_isolated'
        arguments.append('--use_multi_threaded_executor')
    else:
        print(f"Warning: component_container type: {container_type} not recognized."
              "Using component_container_mt.")
        container_executable = 'component_container_mt'
    arguments.extend(['--ros-args', '--log-level', log_level])
    print(f"Using container type: {container_executable}, with arguments: {arguments}")
    return Node(
        name=container_name,
        package='rclcpp_components',
        executable=container_executable,
        on_exit=Shutdown(),
        prefix=prefix,
        arguments=arguments,
        output='screen',
        condition=condition)


def service_call(
        service: str,
        type: str,  # pylint: disable=redefined-builtin
        content: str,
        delay: float | None = None) -> Action:
    """ Add a service call to a launch graph. """
    actions: list[Action] = []
    actions.append(
        ExecuteProcess(
            cmd=[FindExecutable(name='ros2'), ' service call ', service, type, content],
            shell=True,
        ))
    actions.append(log_info(['Calling service ', service, '.']))

    action = GroupAction(actions=actions)
    action = _add_delay_if_set(action, delay)
    return action


def perform_context(context: LaunchContext, expression: Any) -> Any:
    """ If the expression is a substitution perform its substitution else just return the expression. """
    if isinstance(expression, (Substitution)):
        return expression.perform(context)
    else:
        return expression


def play_rosbag(bag_path: Any,
                clock: Any = True,
                loop: Any = None,
                rate: Any = None,
                delay: Any = None,
                shutdown_on_exit: bool = False,
                additional_bag_play_args: Any = None,
                condition: Substitution = None) -> Action:
    """ Add a process playing back a ros2bag to the launch graph. """

    def impl(context: LaunchContext) -> Action:
        bag_path_str = perform_context(context, bag_path)
        loop_str = perform_context(context, loop)
        clock_str = perform_context(context, clock)
        rate_str = perform_context(context, rate)
        delay_str = perform_context(context, delay)
        bag_args_str = perform_context(context, additional_bag_play_args)

        assert pathlib.Path(bag_path_str).exists(), \
            f'[play_rosbag]: Path of rosbag {bag_path_str} does not exist.'

        cmd = f'ros2 bag play {bag_path_str}'.split()
        if is_valid(loop_str) and is_true(loop_str):
            cmd.append('--loop')
        if is_valid(clock_str) and is_true(clock_str):
            cmd.append('--clock')
        if is_valid(rate_str):
            cmd.extend(['--rate', rate_str])
        if is_valid(bag_args_str):
            cmd.extend((bag_args_str).split())

        print("[play_rosbag]: Running the following command:", ' '.join(cmd))
        on_exit_func = None
        if shutdown_on_exit:
            on_exit_func = Shutdown()
        bag_play_action = ExecuteProcess(cmd=cmd, output='screen', on_exit=on_exit_func)
        return [_add_delay_if_set(bag_play_action, delay_str)]

    return OpaqueFunction(function=impl, condition=condition)


def record_rosbag(topics: Any = '--all',
                  delay: Any = None,
                  bag_path: Any = None,
                  additional_bag_record_args: Any = None,
                  storage='mcap',
                  condition: Substitution = None) -> Action:
    """ Add a process recording a ros2bag to the launch graph. """

    def impl(context: LaunchContext) -> Action:
        topics_str = perform_context(context, topics)
        delay_str = perform_context(context, delay)
        bag_path_str = perform_context(context, bag_path)
        bag_args_str = perform_context(context, additional_bag_record_args)

        cmd = f'ros2 bag record --storage {storage} {topics_str}'.split()

        if is_valid(bag_path_str):
            path = pathlib.Path(bag_path_str)
            assert not path.exists(), \
                f'[record_rosbag]: Path of output folder {path} already exist.'
            assert path.parent.exists(), \
                f'[record_rosbag]: Parent path of output folder {path.parent} does not exist.'
            cmd.extend(['--output', bag_path_str])

        if is_valid(bag_args_str):
            cmd.extend((bag_args_str).split())

        print("[record_rosbag]: Running the following command:", ' '.join(cmd))

        bag_play_action = ExecuteProcess(cmd=cmd, output='screen')
        return [_add_delay_if_set(bag_play_action, delay_str)]

    return OpaqueFunction(function=impl, condition=condition)


def static_transform(parent: str,
                     child: str,
                     translation: list[float] | None = None,
                     orientation_rpy: list[float] | None = None,
                     orientation_quaternion: list[float] | None = None,
                     condition=None):
    if translation is None:
        translation = [0, 0, 0]
    if orientation_rpy is None:
        orientation_rpy = [0, 0, 0]

    orientation = orientation_quaternion if orientation_quaternion is not None else orientation_rpy

    translation = [str(x) for x in translation]
    orientation = [str(x) for x in orientation]

    return Node(
        package='tf2_ros',
        name='my_stat_tf_pub',
        executable='static_transform_publisher',
        output='screen',
        arguments=translation + orientation + [parent, child],
        condition=condition,
    )


def shutdown_if_stderr(action: Action) -> Action:
    """ Stop the app if the passed actions prints to stderr. """

    def handler(event) -> Action:
        # pylint: disable=protected-access
        source_action = event._RunningProcessEvent__action
        error = event._ProcessIO__text
        reason = f"Action '{source_action}' failed with error '{error}'."
        print('Shutdown reason:', reason)
        return Shutdown(reason=reason)

    return RegisterEventHandler(OnProcessIO(
        target_action=action,
        on_stderr=handler,
    ))


def set_parameter(parameter: str, value: str, namespace='', condition=None):
    if not namespace:
        return SetParameter(parameter, value, condition=condition)

    yaml_dict = {namespace: {'ros__parameters': {parameter: value}}}
    path = pathlib.Path(f'/tmp/{uuid.uuid4()}.yaml')
    print(f"Writing parameter '{namespace}/{parameter}' to path '{path}'.")
    with open(path, 'w') as file:
        yaml.dump(yaml_dict, file)

    return SetParametersFromFile(str(path), condition=condition)


def has_substring(expression: Any, substring: str) -> bool | Substitution:
    """
    A condition that's true if the expression contains a substring.
    Returns a substitution if the expression is a substitution else returns a boolean.
    """
    if isinstance(expression, (Substitution)):
        return PythonExpression(['"', str(substring), '" in "', expression, '"'])
    else:
        return str(substring) in expression


def is_not(expression: Any) -> bool | Substitution:
    """
    Inverts and expression.
    Returns a substitution if the expression is a substitution else returns a boolean.
    """
    if isinstance(expression, (Substitution)):
        return NotSubstitution(expression)
    else:
        return not expression


def is_empty(expression: Any) -> bool | Substitution:
    """
    Checks if the expression is empty.
    Returns a substitution if the expression is a substitution else returns a boolean.
    """
    if isinstance(expression, (Substitution)):
        return PythonExpression(['len("', expression, '") == 0'])
    else:
        return len(str(expression)) == 0


def is_not_empty(expression: Any) -> bool | Substitution:
    """
    Deprecated: Use `NotSubstitution(is_empty(...))` instead.

    A substitution that's true if the expression is not empty.
    """
    return NotSubstitution(is_empty(expression))


def is_none_or_null(expression: Any) -> bool | Substitution:
    """
    Checks if the expression is 'null' or 'none' or 'False'.
    Returns a substitution if the expression is a substitution else returns a boolean.
    """
    if isinstance(expression, (Substitution)):
        is_none = PythonExpression(['"', expression, '".lower() == "none"'])
        is_null = PythonExpression(['"', expression, '".lower() == "null"'])
        return OrSubstitution(is_none, is_null)
    else:
        return expression is None or str(expression).lower() in ['none', 'null']


def is_true(expression: Any) -> bool | Substitution:
    """
    Checks if the expression is true.
    Returns a substitution if the expression is a substitution else returns a boolean.
    """
    if isinstance(expression, (Substitution)):
        return PythonExpression(['"', expression, '".lower() == "true"'])
    elif isinstance(expression, str):
        return expression.lower() == 'true'
    else:
        return bool(expression)


def is_false(expression: Any) -> bool | Substitution:
    """
    Checks if the expression is false.
    Returns a substitution if the expression is a substitution else returns a boolean.
    """
    if isinstance(expression, (Substitution)):
        return PythonExpression(['"', expression, '".lower() == "false"'])
    elif isinstance(expression, str):
        return expression.lower() == 'false'
    else:
        return not bool(expression)


def is_valid(expression: Any) -> bool | Substitution:
    """
    Checks if the expression is valid.
    We define a valid expression as not being empty and not being null or none.
    Returns a substitution if the expression is a substitution else returns a boolean.
    """
    if isinstance(expression, (Substitution)):
        return AndSubstitution(
            AndSubstitution(
                NotSubstitution(is_none_or_null(expression)),
                NotSubstitution(is_empty(expression)),
            ),
            NotSubstitution(is_false(expression)),
        )
    else:
        return not is_none_or_null(expression) and not is_empty(expression) and not is_false(
            expression)


def is_equal(lhs: Any, rhs: Any) -> bool | Substitution:
    """
    Checks if the two expressions are equal.
    """
    if isinstance(lhs, Substitution) or isinstance(rhs, Substitution):
        return PythonExpression(["'", lhs, "' == '", rhs, "'"])
    else:
        return lhs == rhs


def both_false(a: Substitution, b: Substitution) -> Substitution:
    """ Return substitution which is true if both arguments are false. """
    return AndSubstitution(is_not(a), is_not(b))


def to_bool(expression: Substitution) -> bool | Substitution:
    """ Returns a substitution which is the argument converted to a bool. """
    return is_true(expression)


def union(a: Any, b: Any) -> Substitution:
    """ Unite the expressions a and b. A and be are expected to contain comma-separated strings. """
    return PythonExpression(["','.join(list(set(('", a, "'+','+'", b, "').split(','))))"])


def if_else_substitution(condition: LaunchConfiguration, if_value: Any, else_value: Any) -> Any:
    """ Return if_value if the condition is true, else it returns else_value. """
    return PythonExpression(
        ['"', if_value, '"if "', condition, '".lower() == "true" else"', else_value, '"'])


def get_dict_value(dictionary: Any, key: Any) -> Any:
    """ Returns the value of the item with the specified key. """
    return PythonExpression(['str(', dictionary, '.get("', key, '"))'])


def dict_values_contain_substring(dictionary: Any, substring: str) -> Any:
    """ A substitution that's true if the dictionary holds a value which contains the substring. """
    return PythonExpression(["'", substring, "' in ','.join(list(", dictionary, ".values()))"])


def get_keys_with_substring_in_value(dictionary: Any, substring: str) -> Any:
    """ Return all keys of the items with a value containing the substring. """
    return PythonExpression([
        "','.join(list(key for key, value in ", dictionary, ".items() if '", substring,
        "' in value))"
    ])


def remove_substring_from_dict_values(dictionary: Any, substring: str) -> Any:
    """ Return a dict with a substring being removed from all values of an input dict. """
    return PythonExpression([
        "{ key: ','.join(filter(lambda x: x != '", substring,
        "', value.split(','))) for key, value in", dictionary, ".items()}"
    ])

def remove_substrings_from_dict_values(dictionary: Any, substrings: str) -> Any:
    """ Return a dict with all substrings being removed from all values of an input dict. """
    for substring in substrings:
        dictionary = remove_substring_from_dict_values(dictionary, substring)
    return dictionary


def assert_path_exists(expression: LaunchConfiguration, condition=None) -> Action:
    """
    A condition that's true if the expression evalutes to 'False' in Python.
    Note that the default UnlessCondition is only true if the expression is 'false' or '0'. This is
    more generic, ie. it would also return true for a None type or an empty string.
    """

    def impl(context: LaunchContext) -> None:
        path = pathlib.Path(expression.perform(context))
        assert path.exists(), f'Path {path} does not exist.'

    return OpaqueFunction(function=impl, condition=condition)


def assert_condition(assert_message: str, condition: Condition) -> Action:
    """
    Asserting the condition and printing the assert message if the condition is false.
    """

    def impl(context: LaunchContext) -> None:
        assert False, assert_message

    return OpaqueFunction(function=impl, condition=condition)


def log_info(msg, condition=None) -> Action:
    """ Helper to create a message that is logged from ros launch. """
    return LogInfo(msg=msg, condition=condition)


def get_nova_system_info(path: str = '/etc/nova/systeminfo.yaml') -> dict:
    """ Get the system info dict created by nova init. """
    pathlib_path = pathlib.Path(path)
    assert pathlib_path.exists(), f'Path {pathlib_path} does not exist.'
    yaml_content = pathlib_path.read_text()
    return yaml.safe_load(yaml_content)


def get_nova_robot(path: str = '/etc/nova/manager_selection') -> NovaRobot:
    """ Get the nova robot name stored in the manager_selection file created by nova init. """
    pathlib_path = pathlib.Path(path)
    if not pathlib_path.exists():
        raise FileNotFoundError(f'[get_nova_robot]: manager selection file ' \
                f'{pathlib_path} does not exist.')
    name = pathlib_path.read_text().strip('\n')
    if name == 'nova-carter':
        print(f'Detected NovaRobot: {NovaRobot.NOVA_CARTER.name}')
        return NovaRobot.NOVA_CARTER
    elif name == 'nova-devkit':
        print(f'Detected NovaRobot: {NovaRobot.NOVA_DEVELOPER_KIT.name}')
        return NovaRobot.NOVA_DEVELOPER_KIT
    elif name == 'nova-benchtop':
        print(f'Detected NovaRobot: {NovaRobot.NOVA_BENCHTOP.name}')
        return NovaRobot.NOVA_BENCHTOP
    else:
        print(f'Detection of NovaRobot failed: {NovaRobot.UNKNOWN.name}')
        return NovaRobot.UNKNOWN


def get_isaac_ros_ws_path() -> str:
    isaac_ros_ws_path = os.environ.get('ISAAC_ROS_WS')
    if isaac_ros_ws_path is None:
        isaac_ros_ws_path = "/workspaces/isaac_ros-dev"
        print(
            f"Warning: Isaac ROS workspace path requested, but environment variable ISAAC_ROS_WS "
            "not set. Returning default path {isaac_ros_ws_path}"
        )
    return isaac_ros_ws_path
