#!/root/catkin_ws/.venv/bin/python


import rospy

import gym
from stable_baselines3 import PPO

from bebop_env import BebopEnv


# Constants
N_STEPS = 150
N_EPISODES = 500


if __name__ == "__main__":
    try:
        # register the environment
        gym.register(
            id="BebopEnv-v0",
            entry_point=BebopEnv,
        )
        env = gym.make(
            "BebopEnv-v0", 
            max_episode_steps=N_STEPS
        )

        model = PPO(
            "MultiInputPolicy",
            env, 
            n_steps=N_STEPS, 
            batch_size=N_STEPS, 
            verbose=1
        )

        model.learn(total_timesteps=N_STEPS * N_EPISODES)
        model.save("ppo_bebop")

        obs, info = env.reset()
        while True:
            action, _state = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        env.close()

    except rospy.ROSInterruptException:
        pass
