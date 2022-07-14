
class DefaultConfig:

    @staticmethod
    def config() -> dict:
        return {
            'tensorboard_log_dir': './tb_logs',
            
            #'nlinks_box2d' or 'acrobot_unity'
            'env_key': 'nlinks_box2d',

            # number of joint/links to be controlled
            'n_links': 2,

            # either 'reach' or 'hold'
            'task': 'reach',

            # one of 'images', 'numeric', 'combined'
            'obs_mode': 'combined',

             # quick workaround, set to `True` to train 'cnn_mlp_img' models
            'image_only': False,

            # whether to normalize observations into [-1, 1] - default: True
            'normalize_obs': True,
                         
            # settings for the NLinks2D environment
            'nlink_env_width': 100,
            'nlink_env_height': 100,
            'nlink_env_scale': 40,

            # max number of timesteps for acrobat_unity env
            'max_ts': 500,

            # ROS, GRPC, or ZMQ, only relevant when using the acrobot_unity environment
            'communication_type' : 'GRPC',

            # port number for communication with the simulator, only relevant when using the acrobot_unity environment
            'port_number': '9091',

            # the ip address of the server
            'ip_address': 'localhost',
            
            'use_input_models': True,

            # input model output dimensions
            'global_input_model_latent_dim': 128,
            'joint_input_model_latent_dim': 32,

            'graph_builder_class': 'GraphBuilder',

            # whether to sample actions from a squashed (tanh) distribution
            'squashed_action_dist': False,
            
            # size of the image features extracted from the imahe encoder
            'image_feature_dim': 128,
            'features_extractor_class': 'TargetPredictionCNN',
            'normalize_images': False,
            
            # whether to optimize the CNN with the RL algorithm (set to False when optimizing it separately)
            'train_cnn_with_rl': False,

            # the path from where to load pretrained weights for the CNN
            'cnn_state_dict_path': None,
            
            # number of epochs when optimizing the auxiliary loss of the CNN
            'aux_noptepochs': 2,
            'aux_learning_rate': 2.5e-4,

            'policy_net_class': 'GCNActor',
            'gn_hidden_dim': 256,
            'gn_n_layers': 2,
            'gn_propagation_steps': 1,

            # whether to use a global controller (i.e. actions from a global state),
            # if `False`, actions are produced on a node-level basis
            'actor_global_avg_pool': True,

            'graph_self_edges': False,

            # edges from each joint to all following joints
            'graph_forward_connections': False,

            # edges from the last joint to all previous joints
            'graph_ee_back_connections': True,

            'value_net_hidden_dim': 64,
            'activation_fn': 'relu',

            # whether to use the velocity as part of the state/observation when learning from numeric representaiton
            'use_velocity': True,

            # solver settings
            'n_cpu': 20,
            'n_ts': 1000000,
            'load_model_file_name': False,
            
            # the number of envs on which the agent is periodically evaluated
            'n_eval_envs': 20,

            # the number of episodes to run on each evaluation env
            'n_eval_episodes': 1,

            # after how many timesteps the agent is evaluated on the evaluation envs
            'eval_frequency': 25000,

            # The number of steps to run for each environment per update (i.e. batch size is n_steps * n_env where n_env
            # is number of environment copies running in parallel) - default: 128
            'n_steps': 256,

            # Number of training minibatches per update.For recurrent policies, the number of environments run in
            # parallel should be a multiple of nminibatches. 4
            'nminibatches': 32,

            # Factor for trade - off of bias vs variance for Generalized Advantage Estimator - default: 0.95
            'lam': 0.95,

            # Discount factor - default: 0.99 - (most common), 0.8 to 0.9997
            'gamma': 0.99,

            # Number of epoch when optimizing the surrogate - 4
            'noptepochs': 4,

            # Entropy coefficient for the loss calculation - multiplied by entropy and added to loss 0.01
            'ent_coef': 0.0,

            # learning rate - can be a function - default 2.5e-4
            'learning_rate': 2.5e-4,
            'linear_decay': True,

            # Clipping parameter, it can be a function - default: 0.2 - can be between 0.1-0.3
            'cliprange': 0.2,

            # Value function coefficient for the loss calculation - default: 0.5
            'vf_coef': 0.5,

            # The maximum value for the gradient clipping - default: 0.5
            'max_grad_norm': 0.5,
        }