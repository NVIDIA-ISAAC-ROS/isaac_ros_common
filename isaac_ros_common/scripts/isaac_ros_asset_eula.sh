#!/bin/bash


# Helper script for handling EULA prompts when installing assets. Should be sourced at the beginning
# of a script that installs a *single* asset.
#
# The script:
#  * Fails if EULA has not yet been displayed.
#  * Displays an EULA notice prompt if "--eula" is given.
#  * Checks for EULA acceptance and prints asset output paths if "--print-install-paths" is given (for cmake integration).
#
# The following variables are expected to be defined when sourcing this script:
#
# ISAAC_ROS_WS:         Path to Workspace root.
# ASSET_NAME:           Name of asset. snake_case is recommended here.
# ASSET_DIR:            Directory to install the asset.
# EULA_URL              Location of the EULA.
# ASSET_INSTALL_PATHS:  Output path of the installed artifact.
#
# A marker file is created in "ASSET_DIR" for persistence between sessions.
# Alternatively, the env variable ISAAC_ROS_ACCEPT_EULA can be set. The env
# variable ISAAC_ROS_SKIP_ASSET_INSTALL can be set to skip the installation
# entirely, e.g. for CI runners that don't have a GPU.
#
# This script also defines a function isaac_ros_common_download_asset that can be used to download
# an artifact from a URL or a cache path. By setting ISAAC_ROS_ASSETS_TEST to a non-empty value, the
# script will verify the offline version of the artifact matches the remote version and exit.
#
# Example:
#   isaac_ros_common_download_asset --url https://example.com/artifact.tar.gz --output-path /tmp/artifact.tar.gz --cache-path /tmp/cache/artifact.tar.gz
#   RESULT=$?
#   if [[ -n ${ISAAC_ROS_ASSETS_TEST} ]]; then
#     exit ${RESULT}
#   fi

usage() {
    # Note that the usage is intended for the script that sources this script.
    echo "Usage: $0 [--show-eula|--print-install-paths|--no-cache]"
    echo "Use this script to download the asset '${ASSET_NAME:?}'."
    echo ""
    echo "Explanation of arguments:"
    echo "  --show-eula           Show the EULA and ask user to accept it"
    echo "  --print-install-paths Print the paths where assets will be installed"
    echo "  --no-cache            Don't cache the assets"
}

# Parse arguments
CACHE=true
while [ "$1" != "" ]; do
    case $1 in
        --help )
            usage
            exit 0
            ;;
        # TODO(lgulich): Deprecated, remove this. Use --show-eula instead.
        --eula )
            SHOW_EULA=true
            ;;
        --show-eula )
            SHOW_EULA=true
            ;;
        --print-install-paths )
            PRINT_INSTALL_PATHS=true
            ;;
        --no-cache )
            CACHE=false
            ;;
        * )
            usage
            exit 1
            ;;
    esac
    shift
done

[[ -z "$ISAAC_ROS_WS" ]] && echo "ERROR: ISAAC_ROS_WS is not set." && exit 1
[[ -z "$ASSET_NAME" ]] && echo "ERROR: ASSET_NAME is not set." && exit 1
[[ -z "$ASSET_DIR" ]] && echo "ERROR: ASSET_DIR is not set." && exit 1
[[ -z "$EULA_URL" ]] && echo "ERROR: EULA_URL is not set." && exit 1
[[ -z "$ASSET_INSTALL_PATHS" ]] && echo "ERROR: ASSET_INSTALL_PATHS is not set." && exit 1
[[ -z "$ISAAC_ROS_WS" ]] && echo "ERROR: ISAAC_ROS_WS is not set." && exit 1

MARKER_FILE="$ASSET_DIR/.eula_accepted"


# Print the install paths s.t. other tools can parse it.
if [[ "$PRINT_INSTALL_PATHS" == "true" ]]; then
    echo -n "$ASSET_INSTALL_PATHS"
    exit 0
fi

if [[ -n "$ISAAC_ROS_SKIP_ASSET_INSTALL" ]]; then
    echo "Skipping asset installation."
    exit 0
fi

# Show the EULA and ask user to accept.
if [ "$SHOW_EULA" == "true" ]; then
    echo -e "\n**** EULA notice for asset: \"$ASSET_NAME\"  ****\n"
    echo -e "By continuing you accept the terms and conditions of the license as covered in the Model EULA, found here:"
    echo -e "  $EULA_URL"
    echo -e "\nDo you accept? [y/n]"

    while true; do
        read yn
        case $yn in
            [Yy]* )
              mkdir -p "$(dirname ${MARKER_FILE})"
              touch "$MARKER_FILE"
              break;;
            [Nn]* )
              exit 1
              ;;
            * ) echo "Please answer yes or no.";;
        esac
    done
fi

# Check that the EULA was accepted and abort if not.
if [ ! -f "$MARKER_FILE" ] && [ -z $ISAAC_ROS_ACCEPT_EULA ]; then
    echo -e "\nERROR: Please run the following command to view and accept the EULA before downloading \"$ASSET_NAME\":"
    echo -e  "  $(realpath $0) --eula"
    echo -e "or set environment variable: ISAAC_ROS_ACCEPT_EULA=1"
    exit 1
fi

# Check if the assets were already download and abort early if they are.
if [ "$CACHE" == "true" ] && ls ${ASSET_INSTALL_PATHS} &> /dev/null; then
    echo "All assets for $ASSET_NAME already exist in $ASSET_DIR. Skipping download."
    exit 0
fi

function isaac_ros_common_download_asset() {
    local URL
    local OUTPUT_PATH
    local CACHE_PATH

    # Parse arguments
    while [ "$1" != "" ]; do
        case $1 in
            --url )
                URL=$2
                shift
                ;;
            --output-path )
                OUTPUT_PATH=$2
                shift
                ;;
            --cache-path )
                CACHE_PATH=$2
                shift
                ;;
            * )
                echo "ERROR: Invalid argument: $1"
                exit 1
                ;;
        esac
        shift
    done

    if [[ -z ${OUTPUT_PATH} ]]; then
        echo "ERROR: --output-path is required."
        exit 1
    fi

    if [[ -z ${URL} && -z ${CACHE_PATH} ]]; then
        echo "ERROR: --url or --cache-path is required."
        exit 1
    fi

    if [[ -n ${ISAAC_ROS_ASSETS_TEST} ]]; then
        # Verify the offline version of the artifact matches the remote version and exit
        if [[ -z ${CACHE_PATH} ]]; then
            echo "Skipping testing since no cache path was provided."
            return 0
        elif [[ ! -f ${CACHE_PATH} ]]; then
            echo "ERROR: Cache path ${CACHE_PATH} does not exist."
            return 1
        fi
        echo "Verifying offline version of the artifact matches the remote version."
        OFFLINE_SHA256=$(sha256sum ${CACHE_PATH} | awk '{print $1}')
        echo "Offline version: ${OFFLINE_SHA256}"
        wget -nv "${URL}" -O "${OUTPUT_PATH}"
        REMOTE_SHA256=$(sha256sum "${OUTPUT_PATH}" | awk '{print $1}') 
        echo "Remote version: ${REMOTE_SHA256}"
        if [[ ${OFFLINE_SHA256} == ${REMOTE_SHA256} ]]; then
            return 0
        else
            echo "ERROR: Offline version of the artifact does not match the remote version."
            return 1
        fi
    fi

    if [[ -n ${CACHE_PATH} && -f ${CACHE_PATH} ]]; then
        # Use an offline copy of the artifact
        echo "Copying artifact from ${CACHE_PATH} to ${OUTPUT_PATH}."
        cp "${CACHE_PATH}" "${OUTPUT_PATH}"
    else
        # Download artifact if no cache path was provided or the cache path does not exist
        echo "Downloading artifact from ${URL} to ${OUTPUT_PATH}."
        wget -nv "${URL}" -O "${OUTPUT_PATH}"
    fi
    return 0
}

# Create the ASSET_DIR to prepare the download.
mkdir -p "${ASSET_DIR}"
