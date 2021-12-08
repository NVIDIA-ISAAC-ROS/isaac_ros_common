#!/bin/bash

# Set workspace dir 
ISAAC_ROS_DEV_DIR="test_ws"

# Setup container names
PLATFORM="$(uname -m)"
BASE_NAME="isaac_ros_dev-$PLATFORM"
CONTAINER_NAME="$BASE_NAME-container"

# Map host's display socket to docker
DOCKER_ARGS+=("-v /tmp/.X11-unix:/tmp/.X11-unix")
DOCKER_ARGS+=("-e DISPLAY")
DOCKER_ARGS+=("-e NVIDIA_VISIBLE_DEVICES=all")
DOCKER_ARGS+=("-e NVIDIA_DRIVER_CAPABILITIES=all")

if [[ $PLATFORM == "aarch64" ]]; then
    DOCKER_ARGS+=("-v /usr/bin/tegrastats:/usr/bin/tegrastats")
    DOCKER_ARGS+=("-v /tmp/argus_socket:/tmp/argus_socket")
    DOCKER_ARGS+=("-v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra")
    DOCKER_ARGS+=("-v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api")
fi

# # Optionally load custom docker arguments from file
# DOCKER_ARGS_FILE="$ROOT/.isaac_ros_dev-dockerargs"
# if [[ -f "$DOCKER_ARGS_FILE" ]]; then
#     print_info "Using additional Docker run arguments from $DOCKER_ARGS_FILE"
#     readarray -t DOCKER_ARGS_FILE_LINES < $DOCKER_ARGS_FILE
#     for arg in "${DOCKER_ARGS_FILE_LINES[@]}"; do
#         DOCKER_ARGS+=($(eval "echo $arg | envsubst"))
#     done
# fi

# Run container from image
print_info "Running $CONTAINER_NAME"
docker run -it --rm \
    --privileged --network host \
    ${DOCKER_ARGS[@]} \
    -v $ISAAC_ROS_DEV_DIR:/workspaces/isaac_ros-dev \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --user="admin" \
    --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
    --workdir /workspaces/isaac_ros-dev \
    $@ \
    $BASE_NAME \
    /bin/bash
