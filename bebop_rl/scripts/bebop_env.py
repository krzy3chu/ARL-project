#!/root/catkin_ws/.venv/bin/python


import random

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose
from mav_msgs.msg import Actuators
from std_msgs.msg import String, Float64MultiArray
from std_srvs.srv import Empty

import gym
import numpy as np
from scipy.spatial.transform import Rotation as R


# Constants
STATE_DIM = (
    13  # position (3), orientation (4), linear velocity (3), angular velocity (3)
)
TARGET_HEIGHT = 2
TARGET_STATE = np.array(
    [0, 0, TARGET_HEIGHT] + 3 * [0] + [1] + 6 * [0], dtype=np.float32
)
MAX_POSITION = 10
MAX_ORIENTATION = 1  # for quaternions
MAX_LINEAR_VELOCITY = 25
MAX_ANGULAR_VELOCITY = 25
MAX_ROTOR_SPEED = 750
TIME_STEP = 0.01
LINEAR_VELOCITY_THRESHOLD = 0.05
ANGULAR_VELOCITY_THRESHOLD = 0.05


class BebopEnv(gym.Env):
    def __init__(self):
        super(BebopEnv, self).__init__()
        rospy.init_node("rl_train", anonymous=True)

        self.agent_state = -np.ones(STATE_DIM, dtype=np.float32)
        self.target_location = -np.ones(STATE_DIM, dtype=np.float32)

        self.observation_space_max = np.array(
            3 * [MAX_POSITION]
            + 4 * [MAX_ORIENTATION]
            + 3 * [MAX_LINEAR_VELOCITY]
            + 3 * [MAX_ANGULAR_VELOCITY],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=(STATE_DIM,),
                    dtype=np.float32,
                ),
                "target": gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=(STATE_DIM,),
                    dtype=np.float32,
                ),
            }
        )

        # actions corresponding to motor commands
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

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

        # Debugging topic
        self.reward_pub = rospy.Publisher("/reward", Float64MultiArray, queue_size=10)

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

    def upright(self, orientation):
        # returns -1 if the agent is upside down, 1 if is upright
        x, y, z, w = orientation
        upright = 1 - 2 * (x**2 + y**2)
        return upright

    def quaternion_to_rpy(self, quaternion):
        r = R.from_quat(quaternion)
        roll, pitch, yaw = r.as_euler("xyz")
        return roll, pitch, yaw

    def _get_obs(self):
        # convert the bebop state to agent location
        self.agent_state = np.array(
            [
                self.bebop_state.pose.position.x,
                self.bebop_state.pose.position.y,
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

        # clip the agent state
        self.agent_state = np.clip(
            self.agent_state, -self.observation_space_max, self.observation_space_max
        )

        return {
            "agent": self.agent_state / self.observation_space_max,
            "target": self.target_location / self.observation_space_max,
        }

    def _get_info(self):
        # get agent state information
        roll, pitch, yaw = self.quaternion_to_rpy(self.agent_state[3:7])
        return {
            "position": self.agent_state[0:3],
            "distance_z": abs(self.agent_state[2] - self.target_location[2]),
            "upright": self.upright(self.agent_state[3:7]),
            "roll": roll,
            "pitch": pitch,
            "linear_velocity": np.linalg.norm(self.agent_state[7:10], ord=2),
            "angular_velocity": np.linalg.norm(self.agent_state[10:], ord=2),
        }

    def reset(self, seed: int = None, options: dict = None):
        # reset the Gazebo world
        super().reset(seed=seed)
        self.reset_world_service()

        # spawn bebop at zero position with zero velocity
        zero_quaternion = [0, 0, 0, 1]
        self.agent_state = np.array(
            3 * [0] + zero_quaternion + 6 * [0], dtype=np.float32
        )
        new_bebop_state = ModelState()
        new_bebop_state.pose.position.z = TARGET_HEIGHT
        new_bebop_state.model_name = "bebop"
        self.set_model_state_service(new_bebop_state)

        # set the target location
        self.target_location = TARGET_STATE

        return self._get_obs(), self._get_info()

    def step(self, action):
        # send the motor commands to the bebop
        motor_cmd = Actuators()
        action = (action + 1) * MAX_ROTOR_SPEED / 2
        action = np.clip(action, 0, MAX_ROTOR_SPEED)
        motor_cmd.angular_velocities = action
        self.motor_cmd_pub.publish(motor_cmd)

        # wait specified time for the action to take effect
        rospy.sleep(TIME_STEP)

        # get the new observation
        observation = self._get_obs()
        info = self._get_info()
        terminated = False
        truncated = False

        # calculate the reward
        reward_position = np.clip(1 - abs(info["position"][2] - TARGET_HEIGHT), -1, 1)
        penalty_upright = np.clip(-(info["upright"] - 1) / 2, 0, 1)
        penalty_roll = np.clip(abs(info["roll"] / np.pi), 0, 1)
        penalty_pitch = np.clip(abs(info["pitch"] / np.pi * 2), 0, 1)
        penalty_angular_velocity = np.clip(
            info["angular_velocity"] / MAX_ANGULAR_VELOCITY, 0, 1
        )
        penalty_linear_velocity = np.clip(
            info["linear_velocity"] / MAX_LINEAR_VELOCITY, 0, 1
        )
        reward_total = (
            reward_position
            # - penalty_upright
            - penalty_roll
            - penalty_pitch
            - penalty_angular_velocity
            - penalty_linear_velocity
        )
        reward_total = np.clip(reward_total, -1, 1)

        # terminate if agent is upside down
        if (
            penalty_pitch > 0.8
            or penalty_roll > 0.8
            or info["position"][2] <= 0.2
            or info["position"][2] >= self.observation_space_max[2]
        ):
            terminated = True
            reward_total = -1

        # publish for debugging
        msg = Float64MultiArray()
        msg.data = [
            reward_position,
            # penalty_upright,
            penalty_roll,
            penalty_pitch,
            penalty_angular_velocity,
            penalty_linear_velocity,
            reward_total,
        ]
        self.reward_pub.publish(msg)

        return observation, reward_total, terminated, truncated, info
