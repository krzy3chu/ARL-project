#!/root/catkin_ws/.venv/bin/python


import random

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose
from mav_msgs.msg import Actuators
from std_msgs.msg import String
from std_srvs.srv import Empty

import gym
import numpy as np


# Constants
STATE_DIM = 11  # position in 'z' (1), orientation (4), linear velocity (3), angular velocity (3)
TARGET_STATE = np.array([5] + 3 * [0] + [1] + 6 * [0], dtype=np.float32)
MAX_SPAWN_POSITION = 5
MAX_POSITION = 10
MAX_ORIENTATION = 1  # for quaternions
MAX_LINEAR_VELOCITY = 20
MAX_ANGULAR_VELOCITY = 20
MAX_ROTOR_SPEED = 1000
TIME_STEP = 0.05
POSITION_THRESHOLD = 0.1
LINEAR_VELOCITY_THRESHOLD = 0.1
ANGULAR_VELOCITY_THRESHOLD = 0.1


class BebopEnv(gym.Env):
    def __init__(self):
        super(BebopEnv, self).__init__()
        rospy.init_node("rl_train", anonymous=True)

        self._agent_state = -np.ones(STATE_DIM, dtype=np.float32)
        self._target_location = -np.ones(STATE_DIM, dtype=np.float32)

        observation_space_max = np.array(
            [MAX_POSITION]
            + 4 * [MAX_ORIENTATION]
            + 3 * [MAX_LINEAR_VELOCITY]
            + 3 * [MAX_ANGULAR_VELOCITY],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=-observation_space_max,
                    high=observation_space_max,
                    dtype=np.float32,
                ),
                "target": gym.spaces.Box(
                    low=-observation_space_max,
                    high=observation_space_max,
                    dtype=np.float32,
                ),
            }
        )

        # actions corresponding to motor commands
        self.action_space = gym.spaces.Box(
            low=-MAX_ROTOR_SPEED, high=MAX_ROTOR_SPEED, shape=(4,), dtype=np.float32
        )

        # ROS integration
        self.bebop_state = ModelState()
        self.bebop_state_sub = rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.bebop_state_callback
        )
        self.motor_cmd_pub = rospy.Publisher(
            "/bebop/command/motors", Actuators, queue_size=10
        )
        self.reset_world_service = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.set_model_state_service = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )

        self.reset_world_service.wait_for_service()
        self.set_model_state_service.wait_for_service()

    def bebop_state_callback(self, data):
        try:
            # update the bebop state
            idx = data.name.index("bebop")
            self.bebop_state.model_name = "bebop"
            self.bebop_state.pose = data.pose[idx]
            self.bebop_state.twist = data.twist[idx]
            self.bebop_state.reference_frame = "world"
        except ValueError:
            rospy.logwarn("Bebop model not found in ModelStates")

    def _get_obs(self):
        # convert the bebop state to agent location
        self._agent_state = np.array(
            [
                self.bebop_state.pose.position.z,
                self.bebop_state.pose.orientation.x,
                self.bebop_state.pose.orientation.y,
                self.bebop_state.pose.orientation.z,
                self.bebop_state.pose.orientation.w,
                self.bebop_state.twist.linear.x,
                self.bebop_state.twist.linear.y,
                self.bebop_state.twist.linear.z,
                self.bebop_state.twist.angular.x,
                self.bebop_state.twist.angular.y,
                self.bebop_state.twist.angular.z,
            ]
        ).astype(np.float32)
        return {"agent": self._agent_state, "target": self._target_location}

    def _get_info(self):
        # get agent state information
        return {
            "z_position": self._agent_state[0],
            "z_distance": abs(self._agent_state[0] - self._target_location[0]),
            "orientation": self._agent_state[1:5],
            "orientation_distance": np.dot(
                self._agent_state[1:5], self._target_location[1:5]
            ),
            "linear_velocity": np.linalg.norm(self._agent_state[5:9], ord=2),
            "angular_velocity": np.linalg.norm(self._agent_state[9:], ord=2),
        }

    def reset(self, seed: int = None, options: dict = None):
        # reset the Gazebo world
        super().reset(seed=seed)
        rospy.loginfo("Resetting world, spawning Bebop at random position")
        self.reset_world_service()

        # spawn bebop at a random position with zero velocity
        random_z = np.random.rand() * 2 * MAX_SPAWN_POSITION
        zero_quaternion = [0, 0, 0, 1]
        self._agent_state = np.array(
            [random_z] + zero_quaternion + 6 * [0], dtype=np.float32
        )
        new_bebop_state = ModelState()
        new_bebop_state.model_name = "bebop"
        new_bebop_state.pose.position.z = self._agent_state[0]
        self.set_model_state_service(new_bebop_state)

        # set the target location
        self._target_location = TARGET_STATE

        return self._get_obs(), self._get_info()

    def step(self, action):
        # send the motor commands to the bebop
        motor_cmd = Actuators()
        motor_cmd.angular_velocities = [a * MAX_ROTOR_SPEED for a in action.tolist()]
        self.motor_cmd_pub.publish(motor_cmd)

        # wait specified time for the action to take effect
        rospy.sleep(TIME_STEP)

        # get the new observation
        observation = self._get_obs()
        info = self._get_info()
        terminated = False
        truncated = False

        # calculate the reward
        reward = 0
        reward -= info["z_distance"]
        reward += (info["orientation_distance"] - 1) * 5
        reward -= info["linear_velocity"]
        reward -= info["angular_velocity"]
        if (
            info["z_distance"] < POSITION_THRESHOLD
            and info["linear_velocity"] < LINEAR_VELOCITY_THRESHOLD
            and info["angular_velocity"] < ANGULAR_VELOCITY_THRESHOLD
        ):
            terminated = True
            rospy.loginfo("Target reached")
            reward += 50

        return observation, reward, terminated, truncated, info
