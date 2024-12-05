#!/bin/bash
set -e

# Source for ROS 1 without global ROS_DISTRO variable
unset ROS_DISTRO
source /opt/ros/melodic/setup.bash
source /usr/share/gazebo/setup.sh

# Set up ROS_MASTER_URI and ROS_IP (adjust to your network configuration if needed)
export ROS_MASTER_URI=http://localhost:11311
export ROS_IP=127.0.0.1

roscore &

# Run the command provided as arguments to the container
exec "$@"

