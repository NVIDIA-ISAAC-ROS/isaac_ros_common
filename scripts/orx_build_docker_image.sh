#!/bin/bash

DEVICES = ["ZED", "REALSENSE_D405", "REALSENSE_D435", "HDMI_INPUT"]
DEPLOYMENTS = ["STANDALONE", "ENCODED", "ENCODER", "DECODER"]

build_orx_docker() {
    device = $1
    deployment = $2
    echo "building orx docker for device ${device} in deployment ${deployment}"

}

# ./run_dev.sh [modules]
# ./colcon_clean
# colcon build --packages-up-to [packages]
# ./docker_deploy.sh [options]
# docker push [image]

