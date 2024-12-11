#!/bin/bash
#
# Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $ROOT/utils/print_color.sh
DOCKER_DIR="${ROOT}/../docker"

function usage() {
    print_info "Usage: ${0##*/}"
    print_info "Copyright (c) 2024, NVIDIA CORPORATION."
}

# Initialize arguments
DOCKER_BUILDKIT=1
IGNORE_COMPOSITE_KEYS=0
ADDITIONAL_BUILD_ARGS=()
ADDITIONAL_DOCKER_ARGS=()
DOCKER_SEARCH_DIRS=(${DOCKER_DIR})
SKIP_REGISTRY_CHECK=0
BASE_DOCKER_REGISTRY_NAMES=("nvcr.io/nvidia/isaac/ros")

# Read and parse config file if exists
#
# CONFIG_DOCKER_SEARCH_DIRS (array, can be empty)

if [[ -f "${ROOT}/.isaac_ros_common-config" ]]; then
    . "${ROOT}/.isaac_ros_common-config"
fi

# Override with config from user home directory if exists
if [[ -f ~/.isaac_ros_common-config ]]; then
    . ~/.isaac_ros_common-config
fi

# Prepend configured docker search dirs
if [ ${#CONFIG_DOCKER_SEARCH_DIRS[@]} -gt 0 ]; then
    for (( i=${#CONFIG_DOCKER_SEARCH_DIRS[@]}-1 ; i>=0 ; i-- )); do

        # If the path is relative, then prefix ROOT to the path
        if [[ "${CONFIG_DOCKER_SEARCH_DIRS[i]}" != /* ]]; then
            CONFIG_DOCKER_SEARCH_DIRS[$i]="${ROOT}/${CONFIG_DOCKER_SEARCH_DIRS[i]}"
        fi
    done

    CONFIG_DOCKER_SEARCH_DIRS+=(${DOCKER_SEARCH_DIRS[@]})
    DOCKER_SEARCH_DIRS=(${CONFIG_DOCKER_SEARCH_DIRS[@]})
fi

# Parse command-line args
VALID_ARGS=$(getopt -o hra:b:c:ki:n:d: --long help,skip_registry_check,build_arg:,base_image:,context_dir:,disable_buildkit,image_key:,image_name:,ignore_composite_keys,docker_arg: -- "$@")
eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    -a | --build_arg)
        ADDITIONAL_BUILD_ARGS+=("$2")
        shift 2
        ;;
    -b | --base_image)
        BASE_IMAGE_NAME="$2"
        shift 2
        ;;
    -c | --context_dir)
        DOCKER_CONTEXT_DIR="$2"
        shift 2
        ;;
    -d | --docker_arg)
        ADDITIONAL_DOCKER_ARGS+=("$2")
        shift 2
        ;;
    -k | --disable_buildkit)
        DOCKER_BUILDKIT=0
        shift
        ;;
    -i | --image_key)
        TARGET_IMAGE_STR="$2"
        shift 2
        ;;
    -n | --image_name)
        TARGET_IMAGE_NAME="$2"
        shift 2
        ;;
    -r | --skip_registry_check)
        SKIP_REGISTRY_CHECK=1
        shift
        ;;
    -y | --ignore_composite_keys)
        IGNORE_COMPOSITE_KEYS=1
        shift
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    --) shift;
        break
        ;;
  esac
done

# Check arguments
if [[ -z "$TARGET_IMAGE_STR" ]]; then
    print_error "Target image not specified with -i/--image_key"
    exit 1
fi

if [[ -z "$TARGET_IMAGE_NAME" ]]; then
    TARGET_IMAGE_NAME="${TARGET_IMAGE_STR//./-}-image"
    print_warning "Target image name not specified, using ${TARGET_IMAGE_NAME}"
fi

if [[ ! -z "$DOCKER_CONTEXT_DIR" ]]; then
    DOCKER_SEARCH_DIRS+=($DOCKER_CONTEXT_DIR)
fi

# Summarize final arguments for script
print_info "Building layered image for key ${TARGET_IMAGE_STR} as ${TARGET_IMAGE_NAME}"
if [[ ! -z "${BASE_IMAGE_NAME}" ]]; then
    print_info "Build image on top of base: |${BASE_IMAGE_NAME}|"
fi

print_info "Using configured docker search paths: ${DOCKER_SEARCH_DIRS[*]}"
if [[ ! -z "${DOCKER_CONTEXT_DIR}" ]]; then
    print_info "Docker context directory for final layer: ${DOCKER_CONTEXT_DIR}"
fi
for BUILD_ARG in "${ADDITIONAL_BUILD_ARGS[@]}"
do
    print_info "Additional build arg: ${BUILD_ARG}"
done
for DOCKER_ARG in "${ADDITIONAL_DOCKER_ARGS[@]}"
do
    print_info "Additional docker arg: ${DOCKER_ARG}"
done
if [[ $DOCKER_BUILDKIT -eq 0 ]]; then
    print_warning "WARNING: Explicitly disabling BuildKit"
fi

if [[ $IGNORE_COMPOSITE_KEYS -eq 1 ]]; then
    print_warning "WARNING: Explicitly disabling matching composite image keys"
fi

if [[ $SKIP_REGISTRY_CHECK -eq 1  ]]; then
    print_warning "WARNING: Skipping remote registry check for prebuilt images"
fi

# Setup on-exit cleanup
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
        # Abort matching composite keys if disabled
        if [[ $IGNORE_COMPOSITE_KEYS -gt 1 && $i -eq 1 ]]; then
            break
        fi

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

# Find pre-built image if available
if [[ $SKIP_REGISTRY_CHECK -eq 0 && -z "${BASE_IMAGE_NAME}" ]]; then
    # Generate the possible base image names to look for from first image key onward
    BASE_IMAGE_FULLNAMES=()
    BASE_DOCKERFILES=()
    BASE_DOCKERFILE_CONTEXT_DIRS=()
    BASE_IMAGE_DOCKERFILES_INDICES=()
    for (( i=${#DOCKERFILES[@]}-1 ; i>=0 ; i-- )); do
        BASE_DOCKERFILES+=(${DOCKERFILES[i]})
        BASE_DOCKERFILE_CONTEXT_DIRS+=(${DOCKERFILE_CONTEXT_DIRS[i]})
        BASE_IMAGE_KEYS=()

        DOCKER_HASH_FILE=$(mktemp)
        ON_EXIT+=("rm -Rf ${DOCKER_HASH_FILE}")

        # Determine hash of all Dockerfiles for this base image name
        for (( j=0 ; j<${#BASE_DOCKERFILES[@]} ; j++ )); do
            BASE_DOCKERFILE=${BASE_DOCKERFILES[j]}
            BASE_DOCKERFILE_CONTEXT_DIR=${BASE_DOCKERFILE_CONTEXT_DIRS[j]}
            LAYER_IMAGE_SUFFIX=${BASE_DOCKERFILE#*"/Dockerfile."}
            BASE_IMAGE_KEYS+=(${LAYER_IMAGE_SUFFIX//./ })

            pushd . >/dev/null
            cd $BASE_DOCKERFILE_CONTEXT_DIR
            BASE_DOCKERFILE="Dockerfile.${LAYER_IMAGE_SUFFIX}"
            md5sum ${BASE_DOCKERFILE} >> $DOCKER_HASH_FILE
            popd >/dev/null
        done
        SOURCE_DOCKERFILE_HASH=($(md5sum $DOCKER_HASH_FILE))

        # Determine base image name
        for (( j=${#BASE_DOCKER_REGISTRY_NAMES[@]}-1 ; j>= 0; j-- )); do
            BASE_DOCKER_REGISTRY_NAME=${BASE_DOCKER_REGISTRY_NAMES[j]}
            BASE_IMAGE_TAG=${BASE_IMAGE_KEYS[*]}
            BASE_IMAGE_TAG=${BASE_IMAGE_TAG// /.}
            BASE_IMAGE_FULLNAME="${BASE_DOCKER_REGISTRY_NAME}:${BASE_IMAGE_TAG//./-}_${SOURCE_DOCKERFILE_HASH}"
            BASE_IMAGE_FULLNAMES+=(${BASE_IMAGE_FULLNAME})

            # Remember which index goes with this base image so we can skip those Dockerfiles
            # if this image exists
            BASE_IMAGE_DOCKERFILES_INDICES+=($i)
        done
    done

    for (( i=${#BASE_IMAGE_FULLNAMES[@]}-1 ; i>=0 ; i-- )); do
        BASE_IMAGE_FULLNAME=${BASE_IMAGE_FULLNAMES[i]}

        # Check if image exists on remote server
        print_info "Checking if base image ${BASE_IMAGE_FULLNAME} exists on remote registry"
        OUTPUT=$(docker manifest inspect ${BASE_IMAGE_FULLNAME} >/dev/null 2>&1 ; echo $?)
        if [[ ${OUTPUT} -eq 0 ]]; then
            BASE_IMAGE_NAME=${BASE_IMAGE_FULLNAME}
            DOCKERFILES=(${DOCKERFILES[@]:0:${BASE_IMAGE_DOCKERFILES_INDICES[i]-1}})
            print_info "Found pre-built base image: ${BASE_IMAGE_FULLNAME}"
            docker pull ${BASE_IMAGE_FULLNAME}
            print_info "Finished pulling pre-built base image: ${BASE_IMAGE_FULLNAME}"
            break
        fi
    done
fi

# Arguments for docker build
BUILD_ARGS+=("--build-arg" "USERNAME="admin"")

if [[ $PLATFORM == "x86_64" ]]; then
    BUILD_ARGS+=("--build-arg" "PLATFORM=amd64")
else
    BUILD_ARGS+=("--build-arg" "PLATFORM=arm64")
fi

for BUILD_ARG in ${ADDITIONAL_BUILD_ARGS[@]}
do
    BUILD_ARGS+=("--build-arg" "${BUILD_ARG}")
done

# Check if GPU is installed
if [[ $PLATFORM == "x86_64" ]]; then
    GPU_ATTACHED=(`nvidia-smi -a | grep "Attached GPUs" || true`)
    if [ -z $GPU_ATTACHED ]; then
        print_warning "No GPU detected! Not setting build args for HAS_GPU"
    else
        BUILD_ARGS+=("--build-arg" "HAS_GPU="true"")
    fi
fi

if [ ${#DOCKERFILES[@]} -gt 0 ]; then
    print_info "Resolved the following ${#DOCKERFILES[@]} Dockerfiles for target image: ${TARGET_IMAGE_STR}"
    for DOCKERFILE in ${DOCKERFILES[@]}; do
        print_info "${DOCKERFILE}"
    done
else
    docker tag ${BASE_IMAGE_NAME} ${TARGET_IMAGE_NAME}
    print_info "Nothing to build, retagged ${BASE_IMAGE_NAME} as ${TARGET_IMAGE_NAME}"
    exit 0
fi

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
     "${ADDITIONAL_DOCKER_ARGS[@]}" \
     $@ \
     ${DOCKER_CONTEXT_ARG}
done
