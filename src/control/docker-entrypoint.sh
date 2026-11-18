#!/bin/bash
set -e

source /opt/ros/$ROS_DISTRO/setup.bash
source /opt/pib-datatypes/install/setup.bash
python3 /control/main.py
