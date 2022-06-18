import gym
from stable_baselines3 import PPO
import os
from snekenv_test import SnekEnv

models_dir = "models/Snake/1655482254"
logdir = "logs"

if not os.path.exists(models_dir):
    	os.makedirs(models_dir)
if not os.path.exists(logdir):
    	os.makedirs(logdir)


TIMESTEP = 10000

env = SnekEnv()
env.reset()

#Loading a saved model
model_path = f"{models_dir}/130000.zip"
model = PPO.load(model_path, env = env)

episodes = 10
for ep in range(episodes):
		obs=env.reset()
		done=False
		while not done:
			env.render()
			action, _states = model.predict(obs) #model.predict also returns a state.
			obs, reward, done,info = env.step(action)

# env.close()


