import gym
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common import logger

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class TargetPredictionCNN(BaseFeaturesExtractor):

  def __init__(
    self,
    observation_space: gym.spaces.Box,
    input_channels: int = 1,
    train: bool = False,
    filename: str = None, 
    features_dim: int = 512,
    **kwargs):

    super(TargetPredictionCNN, self).__init__(observation_space, features_dim)

    self.to_grayscale = T.Grayscale()
    self.normalize = T.Normalize(
      mean=kwargs['norm_mean'], 
      std=kwargs['norm_std']
    )

    self.conv1 = nn.Conv2d(input_channels, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(7744, features_dim)
    self.fc2 = nn.Linear(features_dim, 84)
    self.fc3 = nn.Linear(84, 2)
    
    self.filename = filename
    self.should_train = train
    self.normalize_input = kwargs.get('normalize_input', True)

    if filename is not None:
      self.load_state_dict(th.load(filename))
      print(f"Loaded CNN state_dict from '{filename}'.")

    self._train(train)

  def _train(self, train: bool):
    self.should_train = train
    for p in self.parameters():
      p.requires_grad = train
    self.train(train)

  def forward(self, x, only_latent: bool = False):
    def _forward(x):
      x = self.to_grayscale(x)

      if self.normalize_input:
        x = self.normalize(x)

      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = th.flatten(x, 1) # flatten all dimensions except batch
      features = F.relu(self.fc1(x))

      if only_latent:
        return None, features

      x = F.relu(self.fc2(features))
      x = th.tanh(self.fc3(x))

      # logger.record_mean('cnn/0/input', th.mean(x, dim=0))
      # logger.record_mean('cnn/1/activations/conv_pool_1', th.mean(x, dim=0))
      # logger.record_mean('cnn/2/activations/conv_pool_2', th.mean(x, dim=0))
      # logger.record_mean('cnn/3/activations/fc_1', th.mean(features, dim=0))
      # logger.record_mean('cnn/4/activations/fc_2', th.mean(x, dim=0))
      # logger.record_mean('cnn/5/activations/output', th.mean(x, dim=0))
      return x, features
    
    if self.should_train:
      return _forward(x)
    else:
      with th.no_grad(): 
        return _forward(x)
