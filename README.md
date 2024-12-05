# Bebop RL

This project is designed to work with the Bebop drone using reinforcement learning techniques.

## Getting Started

Build the docker image:
```
./build.sh
```

Run the container:
```
./run.sh
```

Build the ROS package:
```
cd /root/catkin_ws
catkin_make --pkg bebop_rl
```

## Running the project

Run Bebop Parrot drone simulator
```
roslaunch bebop_simulator task1_world.launch 
```

Run the project package
```
rosrun bebop_rl rl_train.py
```