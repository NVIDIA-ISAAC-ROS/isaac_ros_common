#!/bin/bash
#
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $ROOT/utils/print_color.sh

BUILDKIT_DISABLED_STR=""
if [[ ! -z "$5" ]]; then
    BUILDKIT_DISABLED_STR="--disable_buildkit"
fi

$ROOT/build_image_layers.sh --image_key "$1" --image_name "$2" --base_image "$3" --context_dir "$4" $BUILDKIT_DISABLED_STR