import optuna
import torch.nn as nn
from typing import Any, Dict

_parameters = {
    'PPO': {
        #'gamma': ('categorical', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
        #'ent_coef': ('log_uniform', [1e-7, 0.1]),
        #'clip_range': ('categorical', [0.1, 0.2, 0.3, 0.4]),
        #'noptepochs': ('categorical', [2, 4, 6]),
        #'max_grad_norm': ('categorical', [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]),
        #'vf_coef': ('uniform', [0.2, 1])
    },
    'optimization': {
        #'n_steps': ('categorical', [128, 256, 512]),
        #'nminibatches': ('categorical', [4, 8, 16, 32]),
        'learning_rate': ('categorical', [0.0025, 0.00025])
        #'linear_decay': ('categorical', [True, False]),
    },
    'architecture': {
        'activation_fn': ('categorical', ["relu", "elu", "leaky_relu"]),
        'value_net_hidden_dim': ('categorical', [64, 128]),
        #'use_input_models': ('categorical', [True, True, False]) # increase odds for true
    },
    'input_models': {
        'global_input_model_latent_dim': ('categorical', [64, 128]),
        'joint_input_model_latent_dim': ('categorical', [16, 32]),
    },
    'GN': {
        'gn_propagation_steps': ('categorical', [3, 4, 5]),
        'gn_n_layers': ('categorical', [1, 2, 3]),
        'gn_hidden_dim': ('categorical', [128, 256, 512])
    },
    'CNN': {
        'image_feature_dim': ('categorical', [64, 128, 256]),
        #'aux_noptepochs': ('categorical', [2, 4]),
        #'aux_learning_rate': ('log_uniform', [1e-4, 0.1]),
    }
}

def _sample(parameters, trial: optuna.Trial) -> Dict[str, Any]:
    sampled_params = {}

    for name, details in parameters.items():
        function, values = details

        if function == 'categorical':
            sampled_params[name] = trial.suggest_categorical(name, values)

        if function == 'uniform':
            sampled_params[name] = trial.suggest_uniform(name, values[0], values[1])

        if function == 'log_uniform':
            sampled_params[name] = trial.suggest_loguniform(name, values[0], values[1])
    
    return sampled_params
        
def sample_ppo_params(trial: optuna.Trial, architecture: str) -> Dict[str, Any]:
    #sampled_params = _sample(_parameters['PPO'], trial)
    sampled_params = _sample(_parameters['optimization'], trial)
    sampled_params.update(_sample(_parameters['architecture'], trial))

    #if sampled_params['use_input_models']:
    sampled_params.update(_sample(_parameters['input_models'], trial))

    if 'GN' in architecture:
        sampled_params.update(_sample(_parameters['GN'], trial))

    if 'CNN' in architecture:
        sampled_params.update(_sample(_parameters['CNN'], trial))

    return sampled_params