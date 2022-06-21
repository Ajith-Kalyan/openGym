from stable_baselines3 import PPO
import os
import time
from carlaenv  import AutonomousCar


models_dir = f"models/Snake/{int(time.time())}/"
logdir = f"logs/Snake/{int(time.time())}/"

if not os.path.exists(models_dir):
    	os.makedirs(models_dir)
if not os.path.exists(logdir):
    	os.makedirs(logdir)

TIMESTEP = 10000

env = AutonomousCar()
env.reset()

#Model creation
model = PPO("MlpPolicy", env, verbose =1, tensorboard_log=logdir)
for i in range(1,10000):
	model.learn(total_timesteps=TIMESTEP, reset_num_timesteps=False,
	 tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEP*i}")

env.close()