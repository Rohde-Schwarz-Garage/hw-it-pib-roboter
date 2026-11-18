# This Dockerfile expects paths relative to the project root.
FROM ros:jazzy-ros-base

# use bash by default
SHELL ["/bin/bash", "-c"]

# update all dependencies and clean up
RUN apt-get update && \
    apt-get full-upgrade -y && \
    rosdep update --rosdistro $ROS_DISTRO && \
    apt-get install python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install datatypes ROS 2 package from https://github.com/pib-rocks/pib-backend/tree/main/ros_packages/datatypes
COPY pib-backend/ros_packages/datatypes /opt/pib-datatypes
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    cd /opt/pib-datatypes && \
    colcon build

RUN pip install tinkerforge --break-system-packages

COPY src/control /control/

WORKDIR /control
CMD [ "/control/docker-entrypoint.sh" ]