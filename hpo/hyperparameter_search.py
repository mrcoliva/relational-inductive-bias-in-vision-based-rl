import os
import json
import time
import optuna
import shutil
import pickle as pkl

from pprint import pprint
from typing import DefaultDict, Type, Dict, Optional, Union
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy

from optuna.integration.skopt import SkoptSampler
from optuna.pruners import BasePruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

from common.callbacks import TrialEvalCallback
from common.agent import Agent
from utils.common import timestamp
from utils.logging import log_run_config
from hpo.utils import sample_ppo_params

# ⚠️ "To run a hyperparameter search with this class, check out 'hpo.py'."

class HyperparameterSearch(object):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        study_name: str,
        default_config: dict,
        sampler_type: str = 'tpe',
        pruner_type: str = 'median',
        n_startup_trials: int = 10,
        n_warmup_steps: int = 20,
        n_trials: int = 10,
        n_eval_envs: int = 10,
        n_eval_episodes: int = 5,
        eval_frequency: int = 1,
        deterministic_eval: bool = True,
        n_jobs: int = 1,
        storage: str = None,
        result_plots: bool = True,
        log_folder = 'log/hpo/',
        verbose: bool = True):

        self.policy = policy
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.default_config = default_config
        self.result_plots = result_plots
        self.verbose = verbose
        self.log_folder = log_folder
        self.n_trials = n_trials
        self.n_eval_envs = n_eval_envs
        self.n_eval_episodes = n_eval_episodes
        self.eval_frequency = eval_frequency
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.storage = storage
        self.sampler_type = sampler_type
        self.pruner_type = pruner_type
        self.deterministic_eval = deterministic_eval

        self.seed = self.default_config['random_seed']
        self.n_timesteps = self.default_config['n_ts']
        self.eval_env = Agent.create_env(self.default_config, n_envs=self.n_eval_envs)

        self.log_folder = os.path.join(
            self.log_folder, 
            self.study_name,
            '02-08_11-59'
        )

    def run(self, study: optuna.Study = None) -> None:
        if study is None:
            os.makedirs(self.log_folder)
            self.write_dict_to_file(vars(self).copy(), 'study.txt')

        set_random_seed(self.seed)

        if self.verbose > 0:
            print(f"\n{'-' * 20} Beginning hyperparameter search {'-' * 20}\n")

        # TODO: eval each hyperparams several times to account for noisy evaluation
        sampler = self._create_sampler(self.sampler_type)
        pruner = self._create_pruner(self.pruner_type)

        if self.verbose > 0:
            print(f"Sampler: {self.sampler_type} - Pruner: {self.pruner_type}")

        self.study = study

        if self.study is None:
            self.study = optuna.create_study(
                sampler=sampler,
                pruner=pruner,
                storage=self.storage,
                study_name=self.study_name,
                load_if_exists=False,
                direction="maximize",
            )

        try:
            self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            pass
        
        # clean up
        # for env in self.eval_env.envs: del env
        
        try:
            self.log_results(study)
        except:
            pass

        if self.result_plots:
            try:
                self.plot_results(study)
            except:
                pass

    def create_agent(self, config: Dict, trial: optuna.Trial):
        sampled_hyperparams = sample_ppo_params(trial, architecture=config['model_architecture'])
        
        config.update(sampled_hyperparams)
        tb_logdir = os.path.join(self.log_folder, 'tb')
        name = f'{self.study_name}_trial_{trial.number}'

        agent = Agent(
            config=config,
            policy=self.policy,
            log_dir=tb_logdir,
            name=name
        )

        return agent, sampled_hyperparams

    def objective(self, trial: optuna.Trial) -> float:
        trial_name = f'trial_{str(trial.number)}'
        path = os.path.join(self.log_folder, trial_name)
        os.makedirs(path, exist_ok=True)
        
        config = self.default_config.copy()
        config['run_start_date'] = timestamp(full=True)

        try:
            agent, sampled_params = self.create_agent(config, trial)
        except Exception as e:
            print('Invalid agent, pruning...', type(e), e.args)
            raise optuna.exceptions.TrialPruned()

        eval_callback = TrialEvalCallback(
            model=agent.model,
            eval_envs=self.eval_env,
            env_seeds=self.default_config['eval_env_seeds'],
            trial=trial,
            frequency=self.eval_frequency,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic_eval,
        )

        try:
            self.write_dict_to_file(
                dict=sampled_params, 
                filename=f'{trial_name}/sampled_params.json', 
                as_json=True)

            config_log_path = os.path.join(path, 'config')
            log_run_config(
                agent=agent,
                filename=config_log_path + '.txt',
                json_filename=config_log_path + '.json')

            agent.train(callbacks=[eval_callback], eval=False, save_model=False)

        except Exception as e:
            print('Error:', type(e), e.args)
            is_pruned = True

        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        self.save_study(self.study)

        # free memory
        try:
            agent.model.env.close()
            del agent.model.env
            del agent.model
            del agent
        except:
            pass
        
        suffix = 'pruned' if is_pruned else 'completed'
        print(f'\nFinished {trial_name}: {suffix}\n')

        if is_pruned:
            #self.rename_tb_log_folder(trial, completed=is_pruned)
            raise optuna.exceptions.TrialPruned()

        return reward

    def handle_exception(self, exception):
        # Sometimes, random hyperparams can generate NaNs or other exceptions
        m = "-" * 25 + ' Trial Aborted ' + '-' * 25
        print(m)
        print('  Reason:', type(exception), exception.args)
        print('-' * len(m))

    def _create_sampler(self, sampler_method: str) -> BaseSampler:
        if sampler_method == "random":
            sampler = RandomSampler(seed=self.seed)
        elif sampler_method == "tpe":
            # TODO: try with multivariate=True
            sampler = TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed)
        elif sampler_method == "skopt":
            # cf https://scikit-optimize.github.io/#skopt.Optimizer
            # GP: gaussian process
            # Gradient boosted regression: GBRT
            sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
        else:
            raise ValueError(f"Unknown sampler: {sampler_method}")
        return sampler

    def _create_pruner(self, pruner_method: str) -> BasePruner:
        # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
        if pruner_method == "halving":
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pruner_method == "median":
            pruner = MedianPruner(
                n_startup_trials=self.n_startup_trials, 
                n_warmup_steps=self.n_warmup_steps,
                n_min_trials=self.n_startup_trials // 2)
        elif pruner_method == "none":
            # Do not prune
            pruner = MedianPruner(n_startup_trials=self.n_trials, n_warmup_steps=self.n_warmup_steps)
        else:
            raise ValueError(f"Unknown pruner: {pruner_method}")
        return pruner

    def write_dict_to_file(self, dict, filename: str, as_json: bool = False):
        path = os.path.join(self.log_folder, filename)

        with open(path, 'w+') as f:
            if as_json:
                f.write(json.dumps(dict, indent=4))
            else:
                [f.write(f'{k}: {str(v)}\n') for k, v in dict.items()]

    def write_report(self, study: optuna.Study, path: str):
        with open(f'{path}.txt', 'w+') as f:
            f.write('=' * 20 + f' Report: {self.study_name} ' + '=' * 20)
            f.write('\n\n')

            f.write(f'Evaluated {len(study.get_trials())} trials.')
            f.write('\n\n')

            f.write('Top 15 trials (last mean rewards):\n' + '-' * 50)
            results = list(map(lambda trial: (trial.number, trial.value), study.get_trials()))
            for trial_number, reward in sorted(results, key=lambda r: r[1], reverse=True)[:15]:
                s = '' if trial_number > 9 else ' '
                f.write(f'\nTrial {trial_number}{s} | {reward:.2f}')
            f.write('\n' + '-' * 50 + '\n\n')

            f.write(f'Best parameters (Trial {str(study.best_trial.number)})\n' + '-' * 50 + '\n')
            f.write(json.dumps(study.best_trial.params, indent=2))
            f.write('\n' + '-' * 50 + '\n')

            f.write('\nStudy:\n' + '-' * 50 + '\n')
            [f.write(f'{k}: {str(v)}\n') for k, v in vars(self).items()]
            f.write('-' * 50)

    def save_study(self, study: optuna.Study):
        path = os.path.join(self.log_folder, f'study_{len(study.get_trials())}_trials')

        with open(f"{path}.pkl", "wb+") as f:
            pkl.dump(study, f)

    def log_results(self, study: optuna.Study):
        print('=' * 25 + ' Results ' + '=' * 25)
        print(f' Number of finished trials:     {len(study.trials)}')
        print(f' Best trial reached reward of:  {study.best_trial.value}')
        
        self.write_dict_to_file(
            dict=study.best_trial.params, 
            filename=f'best_params_(trial_{study.best_trial.number}).json', 
            as_json=True)

        log_path = os.path.join(self.log_folder, 'study')

        if self.verbose:
            print(f"Writing report and study to {log_path}")

        self.write_report(study, log_path)

        # Write report
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        study.trials_dataframe().to_csv(f"{log_path}.csv")

        self.save_study(study)

    def plot_results(self, study: optuna.Study):
        try:
            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)

            fig1.savefig(os.path.join(self.log_folder, 'optimization_history.png'))
            fig2.savefig(os.path.join(self.log_folder, 'param_importances.png'))
        except (ValueError, ImportError, RuntimeError) as e:
            print(f'Error while generating result plots:', e)

    def rename_tb_log_folder(self, trial: optuna.Trial, completed: bool):
        try:
            dir = os.path.join(
                self.log_folder, 'tb', f'{self.study_name}_trial_{trial.number}_1'
            )

            suffix = 'completed' if completed else 'pruned'

            os.rename(dir, dir[:-2] + '_' + suffix)

            #shutil.rmtree(dir)
        except OSError as e:
            print("Failed to delete logs of pruned trial: %s - %s." % (e.filename, e.strerror))