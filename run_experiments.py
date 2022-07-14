# set matplotlib cache dir to a writable directory
import os
os.environ['MPLCONFIGDIR'] = os.path.dirname(os.path.abspath(__file__)) + '/mplconfig'

from utils.common import timestamp
from common import experiment
from torch.multiprocessing import Process, set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass

LOG_FILE_PATH = 'log/experiments/log.txt'

simple_env_config = dict(
    env_key='nlinks_box2d', 
    n_links=2, 
    normalize_images=True
)

complex_env_config = dict(
    env_key='acrobot_unity', 
    n_links=6,
    normalize_images=False, 
    value_net_hidden_dim=128
)

# evaluate every model multiple times
n_runs = 1

# each training run (per model) has a different seed
run_seeds = [200994, 80294, 1704, 2019, 10025][:n_runs]

# seeds of the 20 evaluation environments
eval_env_seeds = [
    30049, 27569, 19874, 30504, 27162, 21079, 12379, 9245, 15690, 24578,
    633, 1661, 20993, 17982, 8426, 29030, 7095, 20715, 6918, 28797
]

def run_experiments():
    args = experiment.argument_parser().parse_args()
    config = experiment.config()

    config['n_eval_envs'] = 20
    config['eval_frequency'] = 25000
    config['eval_env_seeds'] = eval_env_seeds

    # to train a GN with node-level actions, uncomment the following line
    # config['actor_global_avg_pool'] = False
    
    # the following list defines which models should be trained sequentially;
    # for valid ids, see the `config_for_id` function below.
    # e.g. now it trains a GN model on the 2 link env.
    experiment_ids = [
        '2_link_gn',
        #'6_link_gn',
        #'6_link_cnn_gn'
    ]
    
    # conduct `n_runs` experiments for each model
    for eid in experiment_ids:
        evaluate(eid, config.copy(), n_runs, args)

# ........................................................................ #

def evaluate(experiment_id, config, runs, args):
    print(f'[{experiment_id}] Appending progress logs to: {LOG_FILE_PATH}')

    config.update(config_for_id(experiment_id))
    log(f'[{experiment_id}] Beginning experiment.')

    for i in range(runs):
        seed = run_seeds[i]
        config['random_seed'] = seed
        config['info'] = f'exp_{seed}'
    
        log(f'[{experiment_id}] Beginning run {i + 1} (seed: {seed}).')

        process = Process(target=experiment.run, args=(config, args,))
        process.start()
        process.join()

        log(f'[{experiment_id}] Completed run {i + 1}.')

    log(f'[{experiment_id}] Completed {runs} runs.\n')

def config_for_id(experiment_id):
    return {
        # 2 links
        '2_link_gn': simple_env_gn(),
        '2_link_mlp': simple_env_mlp(),
        '2_link_cnn_mlp': simple_env_cnn_mlp(),
        '2_link_cnn_gn': simple_env_cnn_gn(),
        '2_link_cnn_mlp_img': simple_env_enn_mlp_img(),
        # 6 links
        '6_link_gn': complex_env_gn(),
        '6_link_mlp': complex_env_mlp(),
        '6_link_cnn_mlp': complex_env_cnn_mlp(),
        '6_link_cnn_gn': complex_env_cnn_gn(),
        '6_link_cnn_mlp_img': complex_env_cnn_mlp_img()
    }[experiment_id]
        
def simple_env_gn():
    config = simple_env_config.copy()
    config['model_architecture'] = 'GN'
    config['obs_mode'] = 'numeric'
    return config

def simple_env_mlp():
    config = simple_env_config.copy()
    config['model_architecture'] = 'MLP'
    config['policy_net_class'] = 'MLP'
    config['obs_mode'] = 'numeric'
    return config

def simple_env_cnn_gn():
    config = simple_env_config.copy()
    config['model_architecture'] = 'CNN-GN'
    config['policy_net_class'] = 'GCNActor'
    config['obs_mode'] = 'combined'
    return config

def simple_env_cnn_mlp():
    config = simple_env_config.copy()
    config['model_architecture'] = 'CNN-MLP'
    config['policy_net_class'] = 'MLP'
    config['obs_mode'] = 'combined'
    return config

def simple_env_enn_mlp_img():
    config = simple_env_config.copy()
    config['model_architecture'] = 'CNN-MLP-IMG'
    config['policy_net_class'] = 'MLP'
    config['obs_mode'] = 'combined'
    config['image_only'] = True
    return config

def complex_env_gn():
    config = complex_env_config.copy()
    config['model_architecture'] = 'GN'
    config['obs_mode'] = 'numeric'
    return config

def complex_env_mlp():
    config = complex_env_config.copy()
    config['model_architecture'] = 'MLP'
    config['policy_net_class'] = 'MLP'
    config['obs_mode'] = 'numeric'
    return config

def complex_env_cnn_gn():
    config = complex_env_config.copy()
    config['model_architecture'] = 'CNN-GN'
    config['obs_mode'] = 'combined'
    return config

def complex_env_cnn_mlp():
    config = complex_env_config.copy()
    config['model_architecture'] = 'CNN-MLP'
    config['policy_net_class'] = 'MLP'
    config['obs_mode'] = 'combined'
    return config

def complex_env_cnn_mlp_img():
    config = complex_env_config.copy()
    config['model_architecture'] = 'CNN-MLP_IMG'
    config['policy_net_class'] = 'MLP'
    config['obs_mode'] = 'combined'
    config['image_only'] = True
    return config
    
def log(message, also_print: bool = True):
    line = f'[{timestamp(full=True)}] ' + message

    with open(LOG_FILE_PATH, 'a+') as f:
        f.write(line + '\n')

    if also_print:
        print(line)

if __name__ == "__main__":
    run_experiments()
