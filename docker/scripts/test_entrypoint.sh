#!/bin/bash

echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

source /opt/ros/${ROS_DISTRO}/setup.bash

chown -R $USER /isaac_ros-dev

colcon build

echo "source /workspaces/isaac_ros-dev/install/setup.bash" >> ~/.bashrc

source /workspaces/isaac_ros-dev/install/setup.bash

rm -rf /workspaces/isaac_ros_dev/test_results

sudo mkdir -p /workspaces/isaac_ros_dev/test_results

echo Starting Tests...

# -vs to show logs
pytest -v --disable-warnings \
    /workspaces/isaac_ros-dev/src/backend_components/backend_ui_server/backend_ui_server/tests/ \
    /workspaces/isaac_ros-dev/src/configurator/tests/ \
    --junitxml=/workspaces/isaac_ros-dev/test_results/test-result.xml \
    --cov=. \
    --cov-report xml:/workspaces/isaac_ros-dev/test_results/coverage-result.xml
