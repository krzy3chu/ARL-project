#!/root/catkin_ws/.venv/bin/python


import rospy

import gym
from stable_baselines3 import PPO


from bebop_env import BebopEnv


# Constants
N_STEPS = 500
N_EPISODES = 100
N_ITERATIONS = 1
MODEL_PATH = "models/bebop_hover"

if __name__ == "__main__":
    try:
        # register the environment
        gym.register(
            id="BebopEnv-v1",
            entry_point=BebopEnv,
        )
        env = gym.make("BebopEnv-v1", max_episode_steps=N_STEPS)

        # load the model if it exists, otherwise create a new one
        try:
            model = PPO.load(
                MODEL_PATH, env=env, n_steps=N_STEPS, batch_size=N_STEPS, verbose=1
            )
            rospy.loginfo("Model loaded successfully from path %s", MODEL_PATH)
        except FileNotFoundError:
            model = PPO(
                "MultiInputPolicy",
                env=env,
                n_steps=N_STEPS,
                batch_size=N_STEPS,
                verbose=1,
            )
            rospy.loginfo("No existing model found, creating new model")

        # training loop
        for i in range(N_ITERATIONS):
            model.learn(total_timesteps=N_STEPS * N_EPISODES, progress_bar=True)
            model.save(MODEL_PATH)
            rospy.loginfo("Model saved as %s", (MODEL_PATH))

        # test the model
        while True:
            done = False
            obs, info = env.reset()
            while not done:
                action, _state = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            env.close()

    except rospy.ROSInterruptException:
        pass
