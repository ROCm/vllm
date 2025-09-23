#!/bin/bash

# Following the guidelines from B200/GB200 bringup to use mknod(2) to create IMEX channels
# This is admin level operation

find_major_number=$(cat /proc/devices | grep nvidia-caps-imex-channels | wc -l)
if [[ $find_major_number -eq 0 ]];
then
	echo "No nvidia-caps-imex-channels device found. Exiting..."
	exit 1
fi
# /dev/nvidia-caps-imex-channels dir should be present before create a char dev under it.
IMEX_DIRECTORY="/dev/nvidia-caps-imex-channels"
if [ ! -d "$IMEX_DIRECTORY" ]; then
	mkdir -p "$IMEX_DIRECTORY" || { echo "Failed to create directory $IMEX_DIRECTORY"; exit 1; }
fi

major_number=$(cat /proc/devices | grep nvidia-caps-imex-channels | cut -d' ' -f1)
create_cmd="sudo mknod /dev/nvidia-caps-imex-channels/channel0 c $major_number 0"
eval $create_cmd
exit_status=$?
if [[ $exit_status -ne 0 ]];
then
	echo "Unable to create IMEX channel0. StatusCode = $exit_status"
	exit 1
fi
