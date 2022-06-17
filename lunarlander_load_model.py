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

#Loading a saved model
models_dir = "models/PPO"
model_path = f"{models_dir}/PPO__2__90000.zip"
model = PPO.load(model_path, env = env)

episodes = 10
for ep in range(episodes):
		obs=env.reset()
		done=False
		while not done:
			env.render()
			action, _ = model.predict(obs) #model.predict also returns a state.
			obs, reward, done,info = env.step(action)

env.close()