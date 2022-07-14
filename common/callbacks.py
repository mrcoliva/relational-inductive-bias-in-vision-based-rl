import torch
import optuna
import numpy as np
import time

from typing import Optional
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from common.cnn_optimizer import TargetPredictionOptimization
from utils.common import timestamp

class EvalException(Exception):
    pass

class Callback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, model, config, verbose=0):
        super(Callback, self).__init__(verbose)
        
        self.model = model
        self.config = config
        self.cnn_optimizer = None

        # the CNN is not trained with the rl algorithm,
        # but instead trained in a supervised fashion to localize
        # the reaching target
        if config['obs_mode'] != 'numeric' and not config['train_cnn_with_rl']:
            self.cnn_optimizer = TargetPredictionOptimization(
                config=config,
                model=model,
                net=model.policy.features_extractor,
                batch_size=model.batch_size 
            )

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # train the cnn (supervised) on the rollout data for target localization
        if self.cnn_optimizer is not None:
            self.cnn_optimizer.train(self.model.rollout_buffer)

        # log the distributions of the actions and observations of the rollout
        self.log_rollout_distributions()

    def log_rollout_distributions(self):
        # get the whole rollout
        for data in self.model.rollout_buffer.get():

            # log information of the individual joints
            for link in range(self.config['n_links']):
                logger.record(
                    key=f'actions/actions_link_{link}', 
                    value=data.actions[:, link], 
                    exclude='stdout')

                logger.record_mean(
                    key=f'actions/mean_action_link_{link}',
                    value=torch.mean(data.actions[:, link]).detach().cpu().numpy(),
                    exclude='stdout')

class EvalCallback(BaseCallback):

    def __init__(
        self, 
        model, 
        envs,
        env_seeds,
        n_eval_episodes: int = 10,
        frequency: int = 1,
        deterministic: bool = True, 
        verbose=0):

        super(EvalCallback, self).__init__(verbose)

        if env_seeds is not None and len(env_seeds) != envs.num_envs:
            raise ValueError(
                'env_seeds must be None or contain the same number of elements as envs'
            )
        
        self.model = model
        self.env = envs
        self.env_seeds = env_seeds
        self.n_eval_episodes = n_eval_episodes
        self.frequency = frequency
        self.deterministic = deterministic
        self.last_mean_reward = -700
        self.iteration = 0
        self.next_eval_step = 0
        
        self.start_prompt = '-' * 6\
            + f' Evaluating agent on {self.env.num_envs} envs x {self.n_eval_episodes} episodes '\
            + '-' * 6
        self.done_prompt = '-' * len(self.start_prompt)

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        print('|\n| Time:', timestamp(full=True), '\n|')

    def _on_step(self) -> bool:
        if self.model.num_timesteps > self.next_eval_step:
            self._eval()
            self.next_eval_step += self.frequency
        
        return True

    def _eval(self):
        print(self.start_prompt)
        t0 = time.time()

        rewards, total_steps = self._calculate_mean_reward(self.model, self.env)

        mean_reward = np.mean(rewards)
        mean_length = total_steps / self.env.num_envs
        self.last_mean_reward = mean_reward

        logger.record('eval/mean_reward', mean_reward)
        logger.record('eval/mean_ep_len', int(mean_length))

        secs = time.time() - t0
        fps = total_steps // secs

        print(f'  Mean episode length:   | {int(mean_length)}')
        print(f'  Mean reward:           | {int(mean_reward)}')
        print(f'\n  Done. ({secs:.2f} sec | {fps} fps)')
        print(self.done_prompt)

    def _calculate_mean_reward(self, model=None, env=None, n_runs=1):
        """
        A function to calculate a mean reward given a model and an environment to test on. It can
        handle environments that have variable number of timesteps (e.g. if terminal condition is reached before max steps)

        :param model: the model to use for prediction
        :param env: the environment to test on (it is expected that VecEnv environment type is provided)
        :param n_runs: how many runs to perform (e.g. usually the VecEnv has X processes where X is number of CPUs), so for
        for more episodes n_runs is used such that n_runs*X episodes will be executed

        :return: a mean reward, calculated as the average of all episode rewards
        """
        episode_rewards = []
        total_steps = 0

        for _ in range(n_runs):
            [env.envs[i].seed(self.env_seeds[i]) for i in range(len(env.envs))]

            obs = env.reset()
            cum_reward = 0
           
            # running_envs is a mask to make each environment in each process run only once in cases of different number
            # of possible timesteps per environment (usually due to early environment solving due to terminal condition
            # other than the maximum number of timesteps). once all environments have completed the run, each environment is
            # considered again
            running_envs = np.ones(env.num_envs, dtype=bool)

            while True:
                action, _ = model.predict(obs, deterministic=True)
                # set the actions to 0 for finished envs (0 usually interpreted as a "do nothing" action)
                action = action * running_envs[:, None]
                obs, rewards, dones, info = env.step(action)
            
                # use the reward per timestep only from the environments that are still running
                cum_reward += (rewards * running_envs)
                total_steps += np.sum(running_envs)

                # update the running envs (sets to 0 the ones that had terminated in this timestep)
                running_envs = np.multiply(running_envs, np.bitwise_not(dones))
            
                if not np.any(running_envs):
                    episode_rewards.append(cum_reward)
                    break

        return episode_rewards, total_steps

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
        self,
        model,
        eval_envs: VecEnv,
        env_seeds,
        trial: optuna.Trial,
        frequency: int = 1,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        verbose: int = 0
    ):

        super(TrialEvalCallback, self).__init__(
            model=model,
            envs=eval_envs,
            env_seeds=env_seeds,
            n_eval_episodes=n_eval_episodes,
            frequency=frequency,
            deterministic=deterministic,
            verbose=verbose
        )

        self.trial = trial
        self.is_pruned = False
        self.rewards = []

    def _eval(self) -> None:
        super()._eval()

        self.trial.report(self.last_mean_reward, self.model.num_timesteps)
        self.rewards.append(self.last_mean_reward)

        if self.model.num_timesteps > 200_000 and np.max(self.rewards[-2:]) < 100:
            self.is_pruned = True
            raise EvalException(f'Two consecutive rewards {self.rewards[-2:]} below threshold 100.')

        self.is_pruned = self.trial.should_prune()
        
    def _on_step(self) -> bool:
        super()._on_step()
        return not self.is_pruned
