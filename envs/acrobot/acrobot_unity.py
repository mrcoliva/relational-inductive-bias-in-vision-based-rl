import os
import cv2
import base64
import numpy as np
import math

from numpy import newaxis
from stable_baselines3.common import logger
from gym.spaces import Box

from .acrobot import Acrobot

class AcrobotUnity(Acrobot):

    def __init__(self, id, config, is_eval_env: bool = False):
        super().__init__(config['max_ts'])
        self.ENV_KEY = "manipulator_environment"
        self.id = id
        self.row_count = math.ceil(math.sqrt(config["n_cpu"]));
        self.reset_counter = 0
        self.reset_state = None
        self.collided = 0
        self.use_images = config["obs_mode"] == 'combined'
        self.use_velocity = config['use_velocity']
        self.simulator_observation = None
        self.num_joints = config["n_links"]
        self.normalize_obs = config['normalize_obs']
        self.is_eval_env = is_eval_env
        
        # whether the robot must only reach the target or hold position
        self.reach_only = config['task'] == 'reach'

        # the max distance that is enough to solve the task
        self.solved_dist = 0.2

        self.action_space = Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        
        p_max = 1.0         # x-, z-coordinate position limit
        y_max = 1.4         # y-coordinate position limit
        q_max = 2.0 * np.pi # joint angle limit (absolute)
        v_max = 1.9         # joint velocity limit (~110 radians)

        self.state_limits = np.zeros(2 + self.num_joints * 5)
        self.state_limits[:2] = p_max
        self.state_limits[2::5] = q_max
        self.state_limits[3::5] = v_max
        self.state_limits[4::5] = p_max
        self.state_limits[5::5] = y_max
        self.state_limits[6::5] = p_max

        if self.use_images:
            self.observation_space = Box( 
                low=-1, high=1, 
                shape=(100, 100, 4), dtype=np.float32)
        else:
            if self.normalize_obs:
                self.observation_space = Box(
                    low=-1., high=1., shape=(len(self.state_limits),))
            else:
                self.observation_space = Box(
                    low=-self.state_limits, high=self.state_limits)
        
    def _random_target_position(self): 
        d_min, d_max = 0.25, 0.9
        d = self.np_random.uniform(low=d_min, high=d_max)
        angle = self.np_random.uniform(low=0, high=2 * np.pi) 
        x = d * np.cos(angle) 
        z = d * np.sin(angle)
 
        return x, 0.01, z # y = 0 -> target is always on the ground

    def reset(self):
        self.ts = 0
        self.collided = 0

        # sample target position
        self.target_x, self.target_y, self.target_z = self._random_target_position()

        # set target orientation to zero
        self.target_d1, self.target_d2, self.target_d3 = 0, 0, 0
        
        # set object state to zero as it's not used atm
        self.object_x, self.object_y, self.object_z = 0, 0, 0
        self.object_d1, self.object_d2, self.object_d3 = 0, 0, 0

        # the joints to be enabled have value 1 and the others have value 0
        self.reset_state = [1] * self.num_joints + [0] * (7 - self.num_joints)

        # init joint angles and velocities to zero (upright position)
        s = np.zeros(14)

        self.reset_state.extend([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6],
            s[7], s[8], s[9], s[10], s[11], s[12], s[13],
            self.target_x, self.target_y, self.target_z,
            90, 0, 0,
            self.object_x, self.object_y, self.object_z,
            90, 0, 0
        ])

        self.reset_counter += 1

        return np.array(self.reset_state)

    def update(self, observation):
        state, distance = self._convert_observation(observation['Observation'])
        self.ts += 1
 
        distance = np.linalg.norm(distance)
        info = {}
        reached_max_ts = self.ts > self.max_ts
        solved = distance < self.solved_dist if self.reach_only else False
 
        punishment = -1 if self.SPARSE_REWARDS else -distance
        reward = 10 if solved else punishment
        done = solved or reached_max_ts
         
        if done:
            domain = 'eval' if self.is_eval_env else 'rollout'
            logger.record_mean(key=domain + '/solved_fraction', value=solved)
 
        obs = self._combine_obs(observation, state) if self.use_images else state
        return obs, reward, done, info

    def _convert_observation(self, simulator_observation):
        s = simulator_observation
        # the last value from the simulator indicates whether the manipulator collided with the floor, available from
        # version 0.6 of the ManipulatorEnvironment
        self.collided = simulator_observation[-1]
        simulator_obs = simulator_observation[:-1]

        # compute end-effector to target distance
        target_x, target_y, target_z, _, _, _ = self._get_target_coordinates(simulator_obs)
        ee_x, ee_y, ee_z, _, _, _ = self._get_end_effector_coordinates(simulator_obs)
        d_ee_target = np.array([target_x - ee_x, target_y - ee_y, target_z - ee_z])

        state = np.zeros(2 + self.num_joints * 5)
        state[:2] = [target_x, target_z]
        state[2::5] = s[:self.num_joints]
        state[3::5] = s[self.num_joints:self.num_joints * 2]
        state[4::5] = self.num_joints * [ee_x]
        state[5::5] = self.num_joints * [ee_y]
        state[6::5] = self.num_joints * [ee_z]

        if self.normalize_obs:
            state /= self.state_limits

        return state, d_ee_target

    def _retrieve_image(self, observation):
        base64_bytes = observation['ImageData'][0].encode('ascii')
        image_bytes = base64.b64decode(base64_bytes)
        image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
        image = np.asarray(image) / 255.0
        return image

    def _combine_obs(self, observation, state):
        """
        Returns combined numeric and image observations in a 4-channel tensor.
        """
        obs = np.zeros((100, 100, 4))
        obs[:,:,:3] = self._retrieve_image(observation)
        obs[:len(state), 0, 3] = state
        return obs

    def _get_end_effector_coordinates(self, state):
        return \
            -(4 * math.floor(self.id / self.row_count) - state[-18]),\
            state[-17],\
            -(4 * (self.id % self.row_count) - state[-16]),\
            state[-15], state[-14], state[-13]

    def _get_object_coordinates(self, state):
        return \
            state[-6], state[-5], state[-4],\
            state[-3], state[-2], state[-1]

    def _get_target_coordinates(self, state):
        return \
            state[-12] - 4 * math.floor(self.id / self.row_count),\
            state[-11],\
            state[-10] - 4 * (self.id % self.row_count),\
            state[-9], state[-8], state[-7]