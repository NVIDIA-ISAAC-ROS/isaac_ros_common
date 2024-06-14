#!/bin/bash -e
#
# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $ROOT/utils/print_color.sh

# Builds image (ARG MODE=deploy) with base image key, then layers any roots or debians you specify,
# then does rosdep install of a ROS WS if you specify one, then layers on suffix image key, and
# sets a default command to run 'ros2 launch {launch package} {launch file}'

# Example: installs two debians, copies three directories (includes ROS_WS/remaps one), runs package and launch file
# ./docker_deploy.sh -i "libnvvpi3,tensorrt" -d /workspaces/isaac_ros-dev/tests -d /home/nvidia/scripts:/home/admin/scripts -w /workspaces/isaac_ros-dev/ros_ws -b "aarch64.ros2_humble" -p "isaac_ros_image_proc" -f "isaac_ros_image_flip.launch.py"

# Read and parse config file if exists
#
# CONFIG_BASE_IMAGE_KEY (string, can be empty)

if [[ -f "${ROOT}/.isaac_ros_common-config" ]]; then
    . "${ROOT}/.isaac_ros_common-config"
fi

# Override with config from user home directory if exists
if [[ -f ~/.isaac_ros_common-config ]]; then
    . ~/.isaac_ros_common-config
fi

INCLUDE_DIRS=()
INCLUDE_TARBALLS=()
SET_LAUNCH_CMD=0
CUSTOM_APT_SOURCES=()
ADDITIONAL_DOCKER_ARGS=()
VALID_ARGS=$(getopt -o w:d:b:n:s:f:p:i:t:a:c: --long ros_ws:,include_dir:,base_image_key:,name:,suffix_image_key:,launch_file:,launch_package:,install_debians:,include_tarball:,custom_apt_source:,docker_arg: -- "$@")
eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    -a | --custom_apt_source)
        CUSTOM_APT_SOURCES+=("$2")
        shift 2
        ;;
    -b | --base_image_key)
        BASE_IMAGE_KEY="$2"
        shift 2
        ;;
    -c | --docker_arg)
        ADDITIONAL_DOCKER_ARGS+=("--docker_arg $2")        
        shift 2        
        ;;
    -d | --include_dir)
        INCLUDE_DIRS+=("$2")
        shift 2
        ;;
    -f | --launch_file)
        LAUNCH_FILE="$2"
        shift 2
        ;;
    -i | --install_debians)
        INSTALL_DEBIANS_CSV="$2"
        shift 2
        ;;
    -n | --name)
        DEPLOY_IMAGE_NAME="$2"
        shift 2
        ;;
    -p | --launch_package)
        LAUNCH_PACKAGE="$2"
        shift 2
        ;;
    -s | --suffix_image_key)
        SUFFIX_IMAGE_KEY="$2"
        shift 2
        ;;
    -t | --include_tarball)
        INCLUDE_TARBALLS+=("$2")
        shift 2
        ;;
    -w | --ros_ws)
        ROS_WS="$2"
        shift 2
        ;;
    --) shift;
        break
        ;;
  esac
done

# Check arguments
PLATFORM="$(uname -m)"

# Check that both launch file and package are set, or neither
if [[ ! -z "${LAUNCH_FILE}${LAUNCH_PACKAGE}" ]]; then
    if [[ -z "${LAUNCH_FILE}" || -z "${LAUNCH_PACKAGE}" ]]; then
        print_error "Launch package (-p/--launch_package) and launch file (-f/--launch_file) must both be specified or both empty."
        exit 1
    fi
fi

if [[ ! -z "${LAUNCH_FILE}" ]]; then
    SET_LAUNCH_CMD=1
fi

if [[ -z "${BASE_IMAGE_KEY}" ]]; then
    BASE_IMAGE_KEY="${PLATFORM}.ros2_humble"
    print_warning "Base image key not specified, assuming $BASE_IMAGE_KEY"
fi

if [[ -z "${DEPLOY_IMAGE_NAME}" ]]; then
    DEPLOY_IMAGE_NAME="isaac_ros_deploy"
    print_warning "Deploy image name not specified, assuming $DEPLOY_IMAGE_NAME"
fi

if [[ ${#CUSTOM_APT_SOURCES[@]} -gt 100 ]]; then
    print_error "Unable to handle more than 100 custom apt sources."
    exit 1
fi

# Always include install directory of ROS workspace
if [[ ! -z "${ROS_WS}" ]]; then
    SYMLINKS_IN_INSTALL_SPACE=$(find "${ROS_WS}/install" -type l)
    if [ -n "${SYMLINKS_IN_INSTALL_SPACE}" ]; then
        print_warning "Found symlinks in install space. Symlinked install spaces are not supported. Please use an isolated or merge install instead. Symlinks:"
	    print_warning $SYMLINKS_IN_INSTALL_SPACE
    fi

    # Resolve ROS_WS_DEST from install directory setup.sh
    ROS_WS_DEST=/workspaces/isaac_ros-dev
    FILE_CONTENT=$(< "${ROS_WS}/install/setup.sh")
    REGEX="_colcon_prefix_chain_sh_COLCON_CURRENT_PREFIX=([^[:space:]]*)"
    if [[ $FILE_CONTENT =~ $REGEX ]]; then
        ROS_WS_DEST="${BASH_REMATCH[1]%/*}"
    fi

    INCLUDE_DIRS+=( "$ROS_WS/install:${ROS_WS_DEST}/install" )
fi

# Always suffix .user to base image
# If the configured key does not have .user, append it last
if [[ $BASE_IMAGE_KEY != *".user"* ]]; then
    BASE_IMAGE_KEY="${BASE_IMAGE_KEY}.user"
fi

# Summarize final arguments for script
print_info "Building deployable image ${DEPLOY_IMAGE_NAME}"
print_info "Base image key: |${BASE_IMAGE_KEY}| / suffix image_key: |${SUFFIX_IMAGE_KEY}|"
if [[ ! -z "${LAUNCH_FILE}" ]]; then
    print_info "Entrypoint to launch ${LAUNCH_PACKAGE}/${LAUNCH_FILE}"
fi
if [[ ! -z "${INSTALL_DEBIANS_CSV}" ]]; then
    print_info "Installing debians: ${INSTALL_DEBIANS_CSV}"
fi
if [[ ! -z "${ROS_WS}" ]]; then
    print_info "Installing ROS workspace at ${ROS_WS} to ${ROS_WS_DEST}"
fi
for INCLUDE_DIR in "${INCLUDE_DIRS[@]}"
do
    print_info "Installing directory: ${INCLUDE_DIR}"
done
for INCLUDE_TARBALL in "${INCLUDE_TARBALLS[@]}"
do
    print_info "Installing tarball: ${INCLUDE_TARBALL}"
done
for CUSTOM_APT_SOURCE in "${CUSTOM_APT_SOURCES[@]}"
do
    print_info "Adding custom apt source: ${CUSTOM_APT_SOURCE}"
done
for DOCKER_ARG in "${ADDITIONAL_DOCKER_ARGS[@]}"
do
    print_info "Additional docker arg: ${DOCKER_ARG}"
done
print_info "Begin building deployable image"

# Setup on-exit cleanup tasks
ON_EXIT=()
function cleanup {
    for command in "${ON_EXIT[@]}"
    do
        $command
    done
}
trap cleanup EXIT

pushd . >/dev/null
cd $ROOT
ON_EXIT+=("popd")

# Setup staging temp directory
TEMP_DIR=`mktemp -d -t isaac_ros_deploy_XXXXXXXX`
ON_EXIT+=("rm -Rf ${TEMP_DIR}")

pushd . >/dev/null
ON_EXIT+=("popd")

cd $TEMP_DIR
cp -f $ROOT/deploy/_Dockerfile.deploy ${TEMP_DIR}/Dockerfile.deploy
cp -f $ROOT/deploy/_Dockerfile.deploy_ws ${TEMP_DIR}/Dockerfile.deploy_ws
cp -f $ROOT/deploy/_deploy-entrypoint.sh ${TEMP_DIR}/deploy-entrypoint.sh

mkdir -p ${TEMP_DIR}/staging

# Stage directories
for INCLUDE_DIR in "${INCLUDE_DIRS[@]}"
do
    SRC_DIR=${INCLUDE_DIR}
    DEST_DIR=${INCLUDE_DIR}
    INCLUDE_DIR_REMAP_ARRAY=(${INCLUDE_DIR//:/ })
    if [[ ${#INCLUDE_DIR_REMAP_ARRAY[@]} -gt 1 ]]; then
        SRC_DIR=${INCLUDE_DIR_REMAP_ARRAY[0]}
        DEST_DIR=${INCLUDE_DIR_REMAP_ARRAY[1]}
    fi

    print_info "Staging $SRC_DIR->$DEST_DIR"
    mkdir -p ${TEMP_DIR}/staging/${DEST_DIR#/}
    rsync -azL ${SRC_DIR}/ ${TEMP_DIR}/staging/${DEST_DIR#/}
done

# Stage tarballs
for INCLUDE_TARBALL in "${INCLUDE_TARBALLS[@]}"
do
    print_info "Staging tarball $INCLUDE_TARBALL"
    tar -xzvf $INCLUDE_TARBALL -C ${TEMP_DIR}/staging
done

# Delete all .git files and credentials
find $TEMP_DIR -type d -name ".git" | xargs -d '\n' rm -rf

BASE_DEPLOY_IMAGE_NAME="${DEPLOY_IMAGE_NAME}-base"
INSTALLED_DEPLOY_IMAGE_NAME="${DEPLOY_IMAGE_NAME}-installed"

# Stage custom apt sources
for (( i=0; i<${#CUSTOM_APT_SOURCES[@]}; i++ )) do
    printf -v APT_SOURCE_IDX "%02d" $i
    CUSTOM_APT_SOURCE_FILE="${TEMP_DIR}/staging/etc/apt/sources.list.d/${APT_SOURCE_IDX}_custom_apt_source"
    mkdir -p ${CUSTOM_APT_SOURCE_FILE%/*}
    echo "${CUSTOM_APT_SOURCES[$i]}" >> ${CUSTOM_APT_SOURCE_FILE}
done

# Build base image
print_info "Building deploy base image: ${BASE_DEPLOY_IMAGE_NAME} with key ${BASE_IMAGE_KEY}"
$ROOT/build_image_layers.sh --image_key "${BASE_IMAGE_KEY}" --image_name "${BASE_DEPLOY_IMAGE_NAME}" --build_arg "MODE=deploy" ${ADDITIONAL_DOCKER_ARGS[@]}

# Install staged files and setup launch command
print_info "Building install image with launch file ${LAUNCH_FILE} in package ${LAUNCH_PACKAGE}"
$ROOT/build_image_layers.sh --image_key "deploy" --image_name "${INSTALLED_DEPLOY_IMAGE_NAME}" --base_image "${BASE_DEPLOY_IMAGE_NAME}" --context_dir "${TEMP_DIR}" \
    --build_arg "MODE=deploy" --build_arg "SET_LAUNCH_CMD=${SET_LAUNCH_CMD}" --build_arg "LAUNCH_FILE=${LAUNCH_FILE}" --build_arg "LAUNCH_PACKAGE=${LAUNCH_PACKAGE}" --build_arg "INSTALL_DEBIANS_CSV=${INSTALL_DEBIANS_CSV}" ${ADDITIONAL_DOCKER_ARGS[@]}

# Optional, if ROS_WS, install rosdeps
if [[ ! -z "${ROS_WS}" ]]; then
    print_info "Building ROS workspace image for path ${ROS_WS}"
    PREVIOUS_STAGE="${INSTALLED_DEPLOY_IMAGE_NAME}"
    INSTALLED_DEPLOY_IMAGE_NAME="${DEPLOY_IMAGE_NAME}-rosws"
    $ROOT/build_image_layers.sh --image_key "deploy_ws" --image_name "${INSTALLED_DEPLOY_IMAGE_NAME}" --base_image "${PREVIOUS_STAGE}" --context_dir "${TEMP_DIR}" --build_arg "MODE=deploy ROS_WS=${ROS_WS_DEST}" ${ADDITIONAL_DOCKER_ARGS[@]}
fi

# Optional, build suffix image if specified
if [[ ! -z "${SUFFIX_IMAGE_KEY}" ]]; then
    print_info "Building suffix deploy image for key ${SUFFIX_IMAGE_KEY}"
    PREVIOUS_STAGE="${INSTALLED_DEPLOY_IMAGE_NAME}"
    INSTALLED_DEPLOY_IMAGE_NAME="${DEPLOY_IMAGE_NAME}-suffix"
    $ROOT/build_image_layers.sh --image_key "SUFFIX_IMAGE_KEY" --image_name "${INSTALLED_DEPLOY_IMAGE_NAME}" --base_image "${PREVIOUS_STAGE}" --build_arg "MODE=deploy" ${ADDITIONAL_DOCKER_ARGS[@]}
fi

# Retag last image
docker tag "${INSTALLED_DEPLOY_IMAGE_NAME}" "${DEPLOY_IMAGE_NAME}"
print_info "DONE, image: ${DEPLOY_IMAGE_NAME}"
