#!/root/catkin_ws/.venv/bin/python


import rospy

import gym
from tqdm import tqdm

from bebop_env import BebopEnv


# Constants
MAX_STEPS = 200
N_EPISODES = 20


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
                # do random action
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            rospy.loginfo(
                f"Episode {episode} finished with reward {reward}"
            )
        env.close()

    except rospy.ROSInterruptException:
        pass
