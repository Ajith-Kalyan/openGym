import gym
import glob
from gym import spaces
import numpy as np
import cv2
import random
import time
import os
import sys
import math
import keras
from collections import deque

import tensorflow as tf

from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard

import keras.backend.tensorflow_backend as backend
from threading import Thread
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Importing the Carla package library for simulation
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import cv2


#Global Variables
IM_WIDTH = 640
IM_HEIGHT = 480

SHOW_PREVIEW = True   
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_SIZE = 1_000
SECONDS_PER_EPOCH = 15
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 16
TRAINING_BATCH_SIZE = MINIBATCH_SIZE//4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"
MEMORY_FRACTION = 0.8


MIN_REWARD = -100


DISCOUNT = 0.99
EPOCHS = 2000
epsilon  = 1
EPSILON_DECAY = 0.95

MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 2 #10

STEER_FORCE = 0.2
CHECKPOINT_LOC = '/checkpoint'


class AutonomousCar(gym.Env):
    DISPLAY_OUTPUT = SHOW_PREVIEW

    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    actor_list = []
    front_camera = None

    #STANDARDS - to keep track of the collisons that occur. 
    collision_history = []
    lane_history = []
    distance = 0.0


    def __init__(self):
        super(AutonomousCar, self).__init__()
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        #processing the image from the camera sensor from the car.
    def process_img(self, image):
        img = np.array(image.raw_data)
        reshaped_img = img.reshape((self.im_height, self.im_width, 4))

        #taking the first 3 values
        rgb_img = reshaped_img[:, :, :3]
        if self.DISPLAY_OUTPUT:
            cv2.imshow("", rgb_img)
            cv2.waitKey(1)
        # if self.DISPLAY_OUTPUT:
        #     cv2.imshow("", rgb_img)
        #     cv2.waitKey(1)
        self.front_camera = rgb_img
        #return rgb_img/255.0 #normalize the image data. (between 0 and 1)
    def collision_data(self, event):
        self.collision_history.append(event)

    def lane_data(self, event):
        self.lane_history.append(event)

    def obstacle_data(self, event):
        self.distance = float(event.distance)
        #print(self.distance)


    def step(self, action):
    
        #get actions from the vehicle.
        if action == 0: #to go left
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer =  -1*STEER_FORCE ))
        elif action ==1:#to go straight
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer =0))
        elif action == 2: #to go right
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer = 1*STEER_FORCE))

        velocity = self.vehicle.get_velocity()
        velocity_metric = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)) #km per hour

        #Reward mechanism
        #if it collides its -200
        #if it is slower  its -1
        #if it maintains speed and doesn't collide +5

        if len(self.collision_history) != 0:
            done = True
            reward = -20

        elif len(self.lane_history) != 0:
            done = False
            reward = -((20*len(self.lane_history))/len(self.lane_history))

        elif velocity_metric<40:
            done = False
            reward = -1
        elif self.distance <2:
            done = False
            reward = -5
        elif self.distance >2:
            done = False
            reward = 20

        elif velocity_metric>40 and len(self.lane_history) == 0:
            done = False
            reward = 50
        elif velocity_metric>40:
            done = False
            reward = 50
        else :
            done = False
            reward = 10
        if self.epoch_start+SECONDS_PER_EPOCH < time.time():
            done = True

    
        return self.front_camera, reward, done, None

        
    def reset(self):
        self.collision_history = []
        self.actor_list = []
        self.lane_history = []
        self.distance = 0.0



        #Spawning an actor - Vehicle
        self.transform = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        #spawning an actor - Camera to vehicle
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")    
        #self.rgb.set_attribute("fov",f"110")    
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)

        #receive data from the camera sensor
        self.sensor.listen(lambda data: self.process_img(data))

        #initially setting up the vehicle.
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        time.sleep(1)


        #adding collision sensor to the vehicle 
        collision_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        #Lane detection sensor to the vehicle.
        lane_sensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_sensor, transform, attach_to=self.vehicle)

        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self.lane_data(event))

        #obstacle distance detection
        obstacle_sensor = self.blueprint_library.find("sensor.other.obstacle")
        self.obstacle_sensor = self.world.spawn_actor(obstacle_sensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.obstacle_sensor)
        #self.obstacle_sensor.listen(lambda event: self.obstacle_data(event))


        #hardcoding epoch time
        self.epoch_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)
        return self.front_camera




