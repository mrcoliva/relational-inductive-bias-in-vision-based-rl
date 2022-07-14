import gym
import numpy as np

def fits_observation_space(obs, space: gym.spaces.Space) -> bool:
    obs = np.asarray(obs)
    bounds_fit = np.all(obs <= space.high) and np.all(obs >= space.low)
    shape_fits = obs.shape == space.sample().shape
    return bounds_fit and shape_fits