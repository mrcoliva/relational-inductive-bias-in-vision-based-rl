from random import seed
from typing import Optional, Type, Union
from utils.common import get_random_seeds
import networks
import utils

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import PPO

from common.callbacks import Callback, EvalCallback
from envs import NLinksBox2D, AcrobotUnity
from envs.acrobot.multiprocessing_env import SimulatorVecEnv
from common.policies import GraphNetworkPolicy

image_normalization_data = {
    'nlinks_box2d': dict(mean=0.885, std=0.14),
    'acrobot_unity': dict(mean=0.0595, std=0.6685)
}

class Agent(object):

    def __init__(
        self,
        config: dict, 
        policy: Union[str, Type[ActorCriticPolicy]],
        log_dir: Optional[str],
        name: str):

        super(Agent).__init__()
                
        self.obs_mode = config['obs_mode']
        self.config = config
        self.policy = policy
        self.log_dir = log_dir
        self.name = name
        self.path = 'models/' + name
        
        self.env = Agent.create_env(self.config)
        self.model = self.get_model(self.env)

    def train(
        self, 
        callbacks = [], 
        eval: bool = True,
        save_model: bool = True):

        if eval:
            eval_env = Agent.create_env(
                self.config, 
                n_envs=self.config['n_eval_envs'], 
                is_eval_env=True
            )

            callbacks.append(EvalCallback(
                model=self.model,
                envs=eval_env,
                env_seeds=self.config['eval_env_seeds'],
                n_eval_episodes=self.config['n_eval_episodes'],
                frequency=self.config['eval_frequency'])
            )

        callbacks.append(Callback(self.model, self.config))
        
        self.model.learn(
            total_timesteps=self.config['n_ts'],
            tb_log_name=self.name,
            callback=callbacks)
    
        if save_model:
            self.model.save(self.path)

        self.model.env.close()

    def get_model(self, env):
        numeric_features_dim = 2 * self.config['n_links']

        if self.obs_mode == 'numeric':
            numeric_features_dim += 2 # target coordinates

        if self.config['env_key'] == 'acrobot_unity':
            numeric_features_dim += 3 * self.config['n_links'] # ee position

        if self.config['image_only']:
            numeric_features_dim = 0
        
        policy_kwargs = {}

        # check for equality since self.policy is a class, _not_ an instance
        if self.policy == GraphNetworkPolicy:
            policy_kwargs = self._gn_policy_kwargs(numeric_features_dim)
        
        return PPO(
            policy=self.policy,
            env=env,
            policy_kwargs=policy_kwargs,
            n_steps=self.config['n_steps'],
            gamma=self.config['gamma'],
            ent_coef=self.config['ent_coef'],
            learning_rate=self.linear_schedule(self.config),
            clip_range=self.config['cliprange'],
            verbose=1,
            tensorboard_log=self.log_dir)

    def linear_schedule(self, params):
        initial_value = float(params['learning_rate'])
        linear_schedule = params['linear_decay']

        def func(progress):
            """Progress will decrease from 1 (beginning) to 0"""
            if not linear_schedule:
                progress = 1
        
            return min(1, (0.2 + progress)) * initial_value
    
        return func

    def _gn_policy_kwargs(self, numeric_features_dim: int):
        graph_builder_class = getattr(utils.graph_builder, self.config['graph_builder_class'])

        kwargs = dict(
            config=self.config,
            numeric_features_dim=numeric_features_dim,
            global_feature_dim=2,
            joint_feature_dim=2 if self.config['env_key'] == 'nlinks_box2d' else 5,
            graph_builder_class=graph_builder_class,
            activation_fn=self.config['activation_fn']
        )

        if self.obs_mode == 'combined':
            cnn_class = getattr(networks, self.config['features_extractor_class'])

            print('Image norm data', image_normalization_data[self.config['env_key']])

            kwargs.update(dict(
                global_feature_dim=self.config['image_feature_dim'],
                features_extractor_class=cnn_class,
                features_extractor_kwargs=dict(
                    train=self.config['train_cnn_with_rl'],
                    filename=self.config['cnn_state_dict_path'],
                    features_dim=self.config['image_feature_dim'],
                    norm_mean=image_normalization_data[self.config['env_key']]['mean'],
                    norm_std=image_normalization_data[self.config['env_key']]['std'],
                    normalize_input=self.config['normalize_images']
                )
            ))

        return kwargs

    @classmethod
    def create_env(cls, params, n_envs: int = None, is_eval_env: bool = False):
        n_envs = n_envs if n_envs is not None else params['n_cpu']

        print(f'Initializing {n_envs} environments...')

        def instance(id=0):
            if params['env_key'] == 'nlinks_box2d':
                return NLinksBox2D(
                    n_links=params['n_links'],
                    obs_mode=params['obs_mode'], 
                    use_velocity=params['use_velocity'],
                    normalize_obs=params['obs_mode'] != 'images',
                    is_eval_env=is_eval_env,
                    viewport_height=params['nlink_env_height'],
                    viewport_width=params['nlink_env_width'],
                    viewport_scale=params['nlink_env_scale'])

            elif params['env_key'] == 'acrobot_unity':
                return AcrobotUnity(id=id, config=params, is_eval_env=is_eval_env)
            else:
                raise ValueError('Unknown env_key')

        env = [instance for _ in range(n_envs)]
        
        if params['env_key'] == 'nlinks_box2d':
            if n_envs == 1 or is_eval_env:
                env = DummyVecEnv(env)
            else:
                env = SubprocVecEnv(env)
        else:
            env = SimulatorVecEnv(env, params)

        return VecMonitor(env)
