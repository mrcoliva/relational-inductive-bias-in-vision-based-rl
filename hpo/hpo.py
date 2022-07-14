import torch

from common.policies import GraphNetworkPolicy
from hpo.hyperparameter_search import HyperparameterSearch
from common import experiment

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    args = experiment.argument_parser().parse_args()
    config = experiment.config()

    architecture = args.a

    if architecture == 'GN':
        config['obs_mode'] = 'numeric'
    elif architecture == 'CNN-GN':
        config['obs_mode'] = 'combined'
    else:
        raise ValueError('Invalid architecture: ', architecture)

    n_eval_envs = 20

    config['model_architecture'] = architecture
    config['n_cpu'] = 20
    config['random_seed'] = 44713
    config['n_ts'] = 800000
    config['eval_env_seeds'] = [
        111962, 62800, 9097, 75174, 255501, 387683, 471507,\
        98446, 283950, 366573, 390501, 216063, 371194, 179968,\
        238134, 251580, 79700, 331849, 168256, 341660
    ]

    study_name = '_'.join([
        str(config['n_links']), 'link',
        config['model_architecture']
    ])

    search = HyperparameterSearch(
        policy=GraphNetworkPolicy,
        study_name=study_name,
        default_config=config,
        result_plots=True,
        n_trials=75,
        n_startup_trials=15,
        n_eval_episodes=1,
        n_eval_envs=n_eval_envs,
        eval_frequency=100000,
        n_warmup_steps=200000
    )

    search.run()