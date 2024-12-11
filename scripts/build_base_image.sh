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

# Initialize variables
BUILDKIT_DISABLED_STR=""
NO_CACHE_STR=""

# Parse command-line arguments
VALID_ARGS=$(getopt -o '' --long disable-buildkit,no-cache,skip_registry_check -- "$@")
if [[ $? -ne 0 ]]; then
    echo "Invalid arguments"
    exit 1
fi

eval set -- "$VALID_ARGS"

# Process the arguments
while [ : ]; do
    case "$1" in
        --disable-buildkit)
            BUILDKIT_DISABLED_STR="--disable_buildkit"
            shift
            ;;
        --no-cache)
            NO_CACHE_STR="-d --no-cache"
            shift
            ;;
        -r | --skip_registry_check)
            SKIP_REGISTRY_CHECK="--skip_registry_check"
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unexpected option: $1"
            exit 1
            ;;
    esac
done

# Ensure there are at least four remaining positional arguments
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 [--disable-buildkit] [--no-cache] <image_key> <image_name> <base_image> <context_dir>"
    exit 1
fi

# Extract positional arguments
IMAGE_KEY="$1"
IMAGE_NAME="$2"
BASE_IMAGE="$3"
CONTEXT_DIR="$4"

# Call the build_image_layers.sh script with the appropriate arguments
$ROOT/build_image_layers.sh --image_key "$IMAGE_KEY" --image_name "$IMAGE_NAME" --base_image "$BASE_IMAGE" --context_dir "$CONTEXT_DIR" $BUILDKIT_DISABLED_STR $NO_CACHE_STR $SKIP_REGISTRY_CHECK