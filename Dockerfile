# Base ROS Melodic image
FROM ros:melodic-ros-base-bionic

# Install dependencies 
RUN useradd -m -s /bin/bash parrot && echo "parrot ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN apt-get update && apt-get install -y \
    nasm \
    libavahi-core-dev \
    libavahi-client-dev \
    yasm \
    libtool-bin \
    libncurses-dev \
    ros-melodic-ros-comm \
    ros-melodic-roscpp-tutorials \
    ros-melodic-rospy-tutorials \
    ros-melodic-cmake-modules \
    ros-melodic-roslint \
    ros-melodic-xacro \
    ros-melodic-joy \
    ros-melodic-octomap-ros \
    ros-melodic-mavlink \
    ros-melodic-control-toolbox \
    ros-melodic-catkin \
    python3-pytest \
    libeigen3-dev \
    python3-tk \
    ffmpeg \
    python3-numpy \
    ros-melodic-teleop-tools \
    ros-melodic-joy \
    ros-melodic-key-teleop \
    ros-melodic-geometry-msgs \
    ros-melodic-std-msgs \
    ros-melodic-image-view \
    ros-melodic-tf2-geometry-msgs \
    python-rosdep python-rosinstall python-rosinstall-generator python-wstool \
    python-catkin-tools build-essential protobuf-compiler libgoogle-glog-dev \
    git wget curl \
    git curl ssh  \
    libavahi-core-dev \
    libavahi-client-dev \
    libeigen3-dev \
    python3-pip \
    python3-setuptools \
    build-essential \
    pkg-config \
    cmake \
    ros-melodic-plotjuggler-ros \
    && rm -rf /var/lib/apt/lists/*

# Install gazebo
RUN apt-get update && apt-get install -y \
    gazebo9 \
    ros-melodic-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

ENV CMAKE_PREFIX_PATH=/root/parrot_arsdk

# Workspace initialization
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && \
    mkdir -p /root/catkin_ws/src && \
    cd /root/catkin_ws && \
    catkin init && \
    cd /root/catkin_ws && \
    cd /root/catkin_ws/src && \
    git clone https://github.com/AutonomyLab/parrot_arsdk.git && \
    git clone -b med18_gazebo9 https://github.com/gsilano/mav_comm.git && \
    git clone -b dev/gazebo9 https://github.com/gsilano/BebopS.git && \
    git clone -b med18_gazebo9 https://github.com/gsilano/rotors_simulator.git && \
    git clone https://github.com/AutonomyLab/bebop_autonomy.git && \
    cd parrot_arsdk && \
    chmod +x script/download_and_strip_arsdk.sh && \
    ./script/download_and_strip_arsdk.sh "

# Setup python3.8 venv
RUN apt-get update && apt-get install -y python3-yaml python3.8 && \
    python3.8 -m pip install virtualenv && \
    cd /root/catkin_ws && \
    python3.8 -m virtualenv .venv && \
    .venv/bin/pip install rospkg catkin_pkg gym stable-baselines3[extra] shimmy scipy

# Modify CMakeLists.txt in BebopS
RUN /bin/bash -c "sed -i 's/hovering_example/hovering_example2/' /root/catkin_ws/src/BebopS/CMakeLists.txt"

# Build workspace
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && \
    cd /root/catkin_ws && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    catkin_make VERBOSE=1 -j1"

# Add sourcing setup.bash to .bashrc
RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

# Copy entrypoint
COPY ./ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

# Setup entrypoint
ENTRYPOINT ["/ros_entrypoint.sh"]
WORKDIR /root/catkin_ws

# Default command
CMD ["bash"]
