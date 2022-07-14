
import argparse
from common.policies import GraphNetworkPolicy
from common.cnn_optimizer import EncoderNotLearningException
import torch
import gc

from utils.logging import log_run_config
from utils.common import *
from common.agent import Agent
from config import agent_config
from stable_baselines3.common.utils import set_random_seed

def argument_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument('-a', type=str)
  parser.add_argument('--cpus', default=20, type=int)
  parser.add_argument('--steps', default=-1, type=int)
  parser.add_argument('--runs', default=1, type=int)
  parser.add_argument('--notb', default=0, action="store_true")
  parser.add_argument('--info', default=None, type=str)
  parser.add_argument('--cnnpath', default=None, type=str)
  return parser

def config() -> dict:
  return agent_config.current

def cleanup(agent: Agent):
  """free memory"""
  try:
      agent.model.env.close()
      del agent.model.env
      del agent.model
      del agent

      gc.collect()
  except:
      pass

def train(
  config,
  policy,
  log_dir,
  experiment_name,
  save_model: bool, 
  eval: bool):

  print('-' * 45)
  print(f'Starting experiment: {experiment_name}')
  print(f'Writing TensorBoard logs to: {log_dir}')

  random_seed = config.setdefault('random_seed', get_random_seeds(n=1)[0])
  set_random_seed(random_seed)
  print('Using seed:', random_seed)

  agent = Agent(
    config=config,
    policy=policy,
    log_dir=log_dir,
    name=experiment_name
  )

  if save_model:
    log_run_config(agent)
  
  try:
    agent.train(save_model=save_model, eval=True)
    cleanup(agent)
  except EncoderNotLearningException:
    cleanup(agent)
    new_seed = get_random_seeds(n=1)[0]
    config['random_seed'] = new_seed
    print('CNN is not learning, restarting experiment with new seed:', new_seed)

    train(config, policy, log_dir, experiment_name, save_model, eval)

def run(config, args, n_runs: int = 1):
  use_cuda = torch.cuda.is_available()

  if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

  log_dir = None if args.notb else config['tensorboard_log_dir']
  save_model = log_dir != None

  config['n_cpu'] = args.cpus

  if args.steps != -1:
    config['n_ts'] = args.steps

  n_links = config['n_links']
  config['device'] = 'cuda' if use_cuda else 'cpu'
  model_id = config.get('model_architecture', '')

  info_prio = [args.info, config.get('info', None), '']
  info = next(item for item in info_prio if item is not None)

  experiment_name = '_'.join([
    f'{n_links}link', model_id, info, timestamp()
  ])

  evaluate = config['n_eval_envs'] > 0
  if evaluate:
    config.setdefault('eval_env_seeds', get_random_seeds(n=config['n_eval_envs']))

  train(config, GraphNetworkPolicy, log_dir, experiment_name, save_model, evaluate)