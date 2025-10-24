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

# Create the ASSET_DIR to prepare the download.
mkdir -p "${ASSET_DIR}"
