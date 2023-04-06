#!/bin/bash -e
#
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $ROOT/utils/print_color.sh
DOCKER_DIR="${ROOT}/../docker"

function usage() {
    print_info "Usage: build_base_image.sh" {target image, period delimited components, required} {target image name, optional} {disable_build boolean, optional}
    print_info "Copyright (c) 2022, NVIDIA CORPORATION."
}

DOCKER_SEARCH_DIRS=(${DOCKER_DIR})

# Read and parse config file if exists
#
# CONFIG_DOCKER_SEARCH_DIRS (array, can be empty)

if [[ -f "${ROOT}/.isaac_ros_common-config" ]]; then
    . "${ROOT}/.isaac_ros_common-config"

    # Prepend configured docker search dirs
    if [ ${#CONFIG_DOCKER_SEARCH_DIRS[@]} -gt 0 ]; then
        for (( i=${#CONFIG_DOCKER_SEARCH_DIRS[@]}-1 ; i>=0 ; i-- )); do
            if [[ "${CONFIG_DOCKER_SEARCH_DIRS[i]}" != '/*'* ]]; then
                CONFIG_DOCKER_SEARCH_DIRS[$i]="${ROOT}/${CONFIG_DOCKER_SEARCH_DIRS[i]}"
            fi
        done

        CONFIG_DOCKER_SEARCH_DIRS+=(${DOCKER_SEARCH_DIRS[@]})
        DOCKER_SEARCH_DIRS=(${CONFIG_DOCKER_SEARCH_DIRS[@]})

        print_info "Using configured docker search paths: ${DOCKER_SEARCH_DIRS[*]}"
    fi
fi

TARGET_IMAGE_STR="$1"
if [[ -z "$TARGET_IMAGE_STR" ]]; then
    print_error "target image not specified"
    exit 1
fi
shift 1

TARGET_IMAGE_NAME="$1"
if [[ -z "$TARGET_IMAGE_NAME" ]]; then
    TARGET_IMAGE_NAME="${TARGET_IMAGE_STR//./-}-image"
    print_warning "Target image name not specified, using ${TARGET_IMAGE_NAME}"
fi
shift 1


BASE_IMAGE_NAME="$1"
if [[ -z "$BASE_IMAGE_NAME" ]]; then
    print_warning "Using base image name not specified, using '${BASE_IMAGE_NAME}'"
fi
shift 1

DOCKER_CONTEXT_DIR="$1"
if [[ -z "$DOCKER_CONTEXT_DIR" ]]; then
    print_warning "Using docker context dir not specified, using Dockerfile directory"
fi
shift 1

DOCKER_BUILDKIT=1
DISABLE_BUILDKIT_STR="$1"
if [[ ! -z "$DISABLE_BUILDKIT_STR" ]]; then
    print_warning "WARNING: Explicitly disabling BuildKit"
    DOCKER_BUILDKIT=0
fi
shift 1

ON_EXIT=()
function cleanup {
    for command in "${ON_EXIT[@]}"
    do
        $command &>/dev/null
    done
}
trap cleanup EXIT

PLATFORM="$(uname -m)"

# Resolve Dockerfiles by matching target image ids to available files
TARGET_IMAGE_IDS=(${TARGET_IMAGE_STR//./ })
IMAGE_IDS=(${TARGET_IMAGE_IDS[@]})

# Loop over components and find largest tail sequences
# For example, a target image id of 'aarch64.jp5.carter.nav' should match
# Dockerfiles with suffixes in the following order:
# ".aarch64.jp5.carter.nav", ".jp5.carter.nav", ".carter.nav", ".nav"
# If the first file found is ".carter.nav", the matching recurses by then
# looking for the preceding components ".aarch64.jp5" in the same manner

DOCKERFILES=()
DOCKERFILE_CONTEXT_DIRS=()
until [ ${#IMAGE_IDS[@]} -le 0 ]; do
    UNMATCHED_ID_COUNT=${#IMAGE_IDS[@]}

    for (( i=0; i<${#IMAGE_IDS[@]}; i++ )) do
        LAYER_IMAGE_IDS=${IMAGE_IDS[@]:i}
        LAYER_IMAGE_SUFFIX="${LAYER_IMAGE_IDS[@]// /.}"

        for DOCKER_SEARCH_DIR in ${DOCKER_SEARCH_DIRS[@]}; do
            DOCKERFILE="${DOCKER_SEARCH_DIR}/Dockerfile.${LAYER_IMAGE_SUFFIX}"

            if [[ -f "${DOCKERFILE}" ]]; then
                DOCKERFILES+=(${DOCKERFILE})
                DOCKERFILE_CONTEXT_DIRS+=(${DOCKER_SEARCH_DIR})
                IMAGE_IDS=(${IMAGE_IDS[@]:0:i})
                break 2
            fi
        done
    done

    if [ ${UNMATCHED_ID_COUNT} -eq ${#IMAGE_IDS[@]} ]; then
        UNMATCHED_IDS=${IMAGE_IDS[@]}
        MATCHED_DOCKERFILES=${DOCKERFILES[@]}
        print_error "Could not resolve Dockerfiles for target image ids: ${UNMATCHED_IDS// /.}"

        if [ ${#DOCKERFILES[@]} -gt 0 ]; then
            print_warning "Partially resolved the following Dockerfiles for target image: ${TARGET_IMAGE_STR}"
            for DOCKERFILE in ${DOCKERFILES[@]}; do
                print_warning "${DOCKERFILE}"
            done
        fi
        exit 1
    fi
done

# Arguments for docker build
BUILD_ARGS+=("--build-arg" "USERNAME="admin"")
BUILD_ARGS+=("--build-arg" "USER_UID=`id -u`")
BUILD_ARGS+=("--build-arg" "USER_GID=`id -g`")
BUILD_ARGS+=("--build-arg" "PLATFORM=$PLATFORM")

# Check if GPU is installed
if [[ $PLATFORM == "x86_64" ]]; then
    if type nvidia-smi &>/dev/null; then
        GPU_ATTACHED=(`nvidia-smi -a | grep "Attached GPUs"`)
        if [ ! -z $GPU_ATTACHED ]; then
            BUILD_ARGS+=("--build-arg" "HAS_GPU="true"")
        fi
    fi
fi

if [[ "$PLATFORM" == "aarch64" ]]; then
    # Make sure the nvidia docker runtime will be used for builds
    DEFAULT_RUNTIME=$(docker info | grep "Default Runtime: nvidia" ; true)
    if [[ -z "$DEFAULT_RUNTIME" ]]; then
        print_error "Default docker runtime is not nvidia!, please make sure the following line is"
        print_error "present in /etc/docker/daemon.json"
        print_error '"default-runtime": "nvidia",'
        print_error ""
        print_error "And then restart the docker daemon"
        exit 1
    fi
fi


print_info "Resolved the following Dockerfiles for target image: ${TARGET_IMAGE_STR}"
for DOCKERFILE in ${DOCKERFILES[@]}; do
    print_info "${DOCKERFILE}"
done

# Build image layers
for (( i=${#DOCKERFILES[@]}-1 ; i>=0 ; i-- )); do
    DOCKERFILE=${DOCKERFILES[i]}
    DOCKERFILE_CONTEXT_DIR=${DOCKERFILE_CONTEXT_DIRS[i]}
    IMAGE_NAME=${DOCKERFILE#*"/Dockerfile."}
    IMAGE_NAME="${IMAGE_NAME//./-}-image"

    # Build the base images in layers first
    BASE_IMAGE_ARG=
    if [ $i -eq $(( ${#DOCKERFILES[@]} - 1 )) ]; then
        if [[ ! -z "${BASE_IMAGE_NAME}" ]] ; then
            BASE_IMAGE_ARG="--build-arg BASE_IMAGE="${BASE_IMAGE_NAME}""
        fi
    fi

    if [ $i -lt $(( ${#DOCKERFILES[@]} - 1 )) ]; then
        BASE_DOCKERFILE=${DOCKERFILES[i+1]}
        BASE_IMAGE_NAME=${BASE_DOCKERFILE#*"/Dockerfile."}
        BASE_IMAGE_NAME="${BASE_IMAGE_NAME//./-}-image"

        BASE_IMAGE_ARG="--build-arg BASE_IMAGE="${BASE_IMAGE_NAME}""
    fi

    # The last image should be the target image name
    # Use docker context dir script arg only for last image
    DOCKER_CONTEXT_ARG=${DOCKERFILE_CONTEXT_DIR}
    if [ $i -eq 0 ]; then
        IMAGE_NAME=${TARGET_IMAGE_NAME}
        if [[ ! -z "${DOCKER_CONTEXT_DIR}" ]]; then
            DOCKER_CONTEXT_ARG=${DOCKER_CONTEXT_DIR}
        fi
    fi

    print_warning "Building ${DOCKERFILE} as image: ${IMAGE_NAME} with base: ${BASE_IMAGE_NAME}"

    DOCKER_BUILDKIT=${DOCKER_BUILDKIT} docker build -f ${DOCKERFILE} \
     --network host \
     -t ${IMAGE_NAME} \
     ${BASE_IMAGE_ARG} \
     "${BUILD_ARGS[@]}" \
     $@ \
     ${DOCKER_CONTEXT_ARG}
done
