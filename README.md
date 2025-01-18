# Bebop RL

## About the project

The core component of the project is a package implementing the reinforcement learning process for drone control. The project also includes a Docker image with a prepared simulation of the Bebop Parrot drone in the Gazebo environment under ROS Melodic. Developed package implements a custom OpenAI Gym environment integrated with ROS interfaces for controlling a Bebop Parrot drone in the simulation. Project uses the PPO method from the Stable-Baselines3 library to train a drone control model.

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

## Result

An example of the control policy in action is shown in the GIF below.
![bebop_flying.gif](bebop_flying.gif)
