import gym
from stable_baselines3 import PPO
import os

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    	os.makedirs(models_dir)
if not os.path.exists(logdir):
    	os.makedirs(logdir)

TIMESTEP = 10000

env = gym.make("LunarLander-v2")
env.reset()

#Model creation
model = PPO("MlpPolicy", env, verbose =1, tensorboard_log=logdir)
for i in range(1,50):
	model.learn(total_timesteps=TIMESTEP, reset_num_timesteps=False,
	 tb_log_name=PPO)
	model.save(f"{models_dir}/PPO__2__{TIMESTEP*i}")

env.close()