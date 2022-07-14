import numpy as np
from random import randint
from datetime import datetime
import pytz

_timezone = pytz.timezone('Europe/Berlin')

def timestamp(full: bool = False):
  format = "%d.%m.%Y, %H:%M:%S" if full else"%d-%m_%H-%M"
  return datetime.now(_timezone).strftime(format)

def n_trainable_parameters(model):
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  return sum([np.prod(p.size()) for p in model_parameters])

def get_random_seeds(n: int, max: int = 32000):
  return [randint(0, max) for _ in range(n)]