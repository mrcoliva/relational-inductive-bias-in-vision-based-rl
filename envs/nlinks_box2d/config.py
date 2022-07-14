import tensorflow as tf


class Config:

    def __init__(self):
        pass

    @staticmethod
    def get_hyper_dict():
        hyper_dict = {

            'custom_hps': True,

            # cart_pole, bipedal_walker(_hc), lunar_lander, acrobot
            'env_key': 'nlinks_box2d',
            'env_para': 2,

            # ROS or ZMQ, only relevant when using the acrobot_unity environment
            'communication_type' : 'ZMQ',

            # port number for communication with the simulator, only relevant when using the acrobot_unity environment
            'port_number': '9090',

            # whether to use image observations for learning:
            'use_images': True,

            # whether to use the velocity as part of the state/observation when learning from numeric representaiton
            'use_velocity': False,

            # solver settings
            'n_cpu': 1,
            'n_ts': 1500000,
            'load_model_file_name': False, #'model_nlinks-4_1.pkl',

            # adjust the policy network pi: policy; vf: value
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])],

            # The number of steps to run for each environment per update (i.e. batch size is n_steps * n_env where n_env
            # is number of environment copies running in parallel) - default: 128
            'n_steps': 2048,

            # Number of training minibatches per update.For recurrent policies, the number of environments run in
            # parallel should be a multiple of nminibatches. 4
            'nminibatches': 32,

            # Factor for trade - off of bias vs variance for Generalized Advantage Estimator - default: 0.95
            'lam': 0.95,

            # Discount factor - default: 0.99 - (most common), 0.8 to 0.9997
            'gamma': 0.99,

            # Number of epoch when optimizing the surrogate - 4
            'noptepochs': 10,

            # Entropy coefficient for the loss calculation - multiplied by entropy and added to loss 0.01
            'ent_coef': 0.0,

            # learning rate - can be a function - default 2.5 e-4
            'learning_rate': 2.5e-3,
            'linear_decay': True, #True

            # Clipping parameter, it can be a function - default: 0.2 - can be between 0.1-0.3
            'cliprange': 0.2,

            # Value function coefficient for the loss calculation - default: 0.5
            'vf_coef': 0.5,

            # The maximum value for the gradient clipping - default: 0.5
            'max_grad_norm': 0.5,
        }

        return hyper_dict

    # rl zoo for inverted pendulum swing up
    # normalize: true
    # n_envs: 8
    # n_timesteps: !!float 2e6
    #######################################
    # policy: 'MlpPolicy'
    # n_steps: 2048
    # nminibatches: 32
    # lam: 0.95
    # gamma: 0.99
    # noptepochs: 10
    # ent_coef: 0.0
    # learning_rate: 2.5e-4
    # cliprange: 0.2
