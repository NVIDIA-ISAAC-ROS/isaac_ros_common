#!/usr/bin/env bash

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# This script is triggered when a corresponding udev rule is matched with action 
function usage() {
	echo "Usage: hotplug-realsense.sh -a <action=add/remove> -d <dev_path> [options]"
	echo "-a | --action     		Action determines whether the device has been added or removed. Valid values are 'add' or 'remove'."
	echo "-d | --dev-path   		Device path that will be used to mount the device."
	echo "-M | --major-version      The kernel major number for the device. This will be used when action=add."
	echo "-m | --minor-version      The kernel minor number for the device. This will be used when action=add."
	echo "-p | --parent-path      	Parent node path of the device pointed by the -d flag."
	echo "-h | --help       		Display this message."
}

function check_mandatory_param() {
VAR=$1
MSG=$2
if [[ -z ${VAR} ]] ; then
	echo  "${MSG}"
	usage
	exit 1
fi
}

function timestamp() {
	echo [$EPOCHREALTIME] [$(date)] 
}

ARGUMENTS=$(getopt -n hotplug-realsense.sh -o a:d:M:m:p:h -l action:,dev-path:,major-version:,minor-version:,parent-path:,help -- "$@" )

if [[ $? -ne 0 ]]; then
	usage
fi

eval set -- "$ARGUMENTS"

while [ : ]; do
  case "$1" in
    -a | --action)
        ACTION=$2 ; shift 2 ;;	
    -d | --dev-path)
        DEV_PATH=$2 ; shift 2 ;;
    -M | --major-version)
        MAJOR_VERSION=$2 ; shift 2 ;;
    -m | --minor-version)
        MINOR_VERSION=$2 ; shift 2 ;;
	-p | --parent-path)
        PARENT_PATH=/dev/$2 ; shift 2 ;;
	-h | --help)
        usage ; shift ;;
    --) shift; break ;;
  esac
done

check_mandatory_param ${ACTION:-""} "Please provide valid value for action" 
check_mandatory_param ${DEV_PATH:-""} "Please provide valid value for device path" 

if [[ "${ACTION}" == "add" ]]; then
	check_mandatory_param ${MAJOR_VERSION:-""} "Please provide valid value for major number" 
	check_mandatory_param ${MINOR_VERSION:-""} "Please provide valid value for minor number" 
	sudo mknod -m a=rw ${DEV_PATH} c ${MAJOR_VERSION} ${MINOR_VERSION}
	sudo chown root:plugdev ${DEV_PATH}
	echo $(timestamp) "Added ${DEV_PATH} with major version: ${MAJOR_VERSION} and minor version: ${MINOR_VERSION} to docker" >> /tmp/docker_usb.log
elif [[ "$ACTION" == "remove" ]]; then
	sudo rm ${DEV_PATH} ${PARENT_PATH}
	echo $(timestamp) "Removed ${DEV_PATH} ${PARENT_PATH} from docker" >> /tmp/docker_usb.log
else
	echo "Cannot recognize action=${ACTION}"
	usage
	exit 1
fi
