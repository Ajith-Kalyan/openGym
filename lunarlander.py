import gym
from stable_baselines3 import PPO

import os


MODEL_NAME = "PPO"
models_dir = "models/{MODEL_NAME}"
logdir = "logs"

if not os.path.exists(models_dir):
    	os.makedirs(models_dir)
if not os.path.exists(logdir):
    	os.makedirs(logdir)


TIMESTEP = 10000
MODEL_NAME = "PPO"

env = gym.make("LunarLander-v2")
env.reset()

#Loading a saved model
models_dir = "models/{MODEL_NAME}"
model_path = f"{models_dir}/PPO__2__490000.zip"
model = MODEL_NAME.load(model_path, env = env)

'''
#Model creation
model = MODEL_NAME("MlpPolicy", env, verbose =1, tensorboard_log=logdir)
for i in range(1,50):
	model.learn(total_timesteps=TIMESTEP, reset_num_timesteps=False,
	 tb_log_name=MODEL_NAME)
	model.save(f"{models_dir}/{MODEL_NAME}__2__{TIMESTEP*i}")
'''

episodes = 10
for ep in range(episodes):
		obs=env.reset()
		done=False
		while not done:
			env.render()
			action, _ = model.predict(obs) #model.predict also returns a state.
			obs, reward, done,info = env.step(action)

env.close()