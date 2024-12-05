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
from tqdm import tqdm


# Constants
MIN_SPAWN_POSITION = -5
MAX_SPAWN_POSITION = 5
MIN_OBSERVE_POSITION = -10
MAX_OBSERVE_POSITION = 10
TARGET_POSITION = np.array([0, 0, 5], dtype=np.float32)
MIN_ROTOR_SPEED = -1000
MAX_ROTOR_SPEED = 1000
TIME_STEP = 0.05
MAX_STEPS = 200
N_EPISODES = 20


class BebopEnv(gym.Env):
    def __init__(self):
        super(BebopEnv, self).__init__()
        rospy.init_node("rl_train", anonymous=True)

        # define the agent and target location
        self._agent_location = np.array([-1, -1, -1], dtype=np.float32)
        self._target_location = np.array([-1, -1, -1], dtype=np.float32)

        # assume using only position as x, y, z
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    MIN_OBSERVE_POSITION,
                    MAX_OBSERVE_POSITION,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "target": gym.spaces.Box(
                    MIN_OBSERVE_POSITION,
                    MAX_OBSERVE_POSITION,
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )

        # actions corresponding to motor commands
        self.action_space = gym.spaces.Box(
            low=MIN_ROTOR_SPEED, high=MAX_ROTOR_SPEED, shape=(4,), dtype=np.float32
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
        # update the bebop state
        try:
            idx = data.name.index("bebop")
            self.bebop_state.model_name = "bebop"
            self.bebop_state.pose = data.pose[idx]
            self.bebop_state.twist = data.twist[idx]
            self.bebop_state.reference_frame = "world"
        except ValueError:
            rospy.logwarn("Bebop model not found in ModelStates")

    def _get_obs(self):
        self._agent_location = np.array(
            [
                self.bebop_state.pose.position.x,
                self.bebop_state.pose.position.y,
                self.bebop_state.pose.position.z,
            ]
        ).astype(np.float32)
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=2
            )
        }

    def reset(self, seed: int = None, options: dict = None):
        # reset the Gazebo world and spawn bebop at a random position
        super().reset(seed=seed)
        rospy.loginfo("Resetting world, spawning Bebop at random position")
        self.reset_world_service()

        position_range = MAX_SPAWN_POSITION - MIN_SPAWN_POSITION
        self._agent_location = np.random.rand(3) * position_range + MIN_SPAWN_POSITION
        new_bebop_state = ModelState()
        new_bebop_state.model_name = "bebop"
        new_bebop_state.pose.position.x = self._agent_location[0]
        new_bebop_state.pose.position.y = self._agent_location[1]
        new_bebop_state.pose.position.z = self._agent_location[2] + MAX_SPAWN_POSITION
        self.set_model_state_service(new_bebop_state)

        self._target_location = TARGET_POSITION

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # send the motor commands to the bebop
        motor_cmd = Actuators()
        motor_cmd.angular_velocities = action.tolist()
        self.motor_cmd_pub.publish(motor_cmd)

        # wait specified time for the action to take effect
        rospy.sleep(TIME_STEP)

        # get the new observation
        observation = self._get_obs()
        info = self._get_info()
        reward = -info["distance"]
        terminated = info["distance"] < 0.1
        truncated = False

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    try:
        # register the environment
        gym.register(
            id="BebopEnv-v0",
            entry_point=BebopEnv,
        )
        env = gym.make(
            "BebopEnv-v0",
            max_episode_steps=MAX_STEPS,
        )

        # run the learning loop
        for episode in tqdm(range(N_EPISODES)):
            obs, info = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            rospy.loginfo(
                f"Episode {episode} finished with precision {info['distance']}"
            )
        env.close()

    except rospy.ROSInterruptException:
        pass
