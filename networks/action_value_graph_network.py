import torch
import torch.nn as nn

from typing import Tuple, Type, Union
from stable_baselines3.common.utils import get_device
from networks.actor_gn import ActorGN, TransformerActor, GCNActor, GatedGCActor, GraphConvActor, LEConvActor, ResGatedGraphConvActor
from networks.critic_gn import CriticGN
from config import agent_config
from utils.graph_builder import GraphBuilder

gn_map = {
    'TransformerActor': TransformerActor,
    'GCNActor': GCNActor,
    'GatedGCActor': GatedGCActor,
    'GraphConvActor': GraphConvActor,
    'LEConvActor': LEConvActor,
    'ResGatedGraphConvActor': ResGatedGraphConvActor
}

class ActionValueGraphNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param activation_fn: the activation function to be used in the action and value networks
    """

    def __init__(
        self,
        graph_builder: GraphBuilder,
        config: dict,
        input_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[torch.device, str] = "auto"
    ):
        super(ActionValueGraphNetwork, self).__init__()
        
        self.graph_builder = graph_builder
        self.image_only = config['image_only']

        action_dim = config['n_links']
        policy_gn_hidden_dim = config['gn_hidden_dim']
        value_net_hidden_dim = config['value_net_hidden_dim']

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = policy_gn_hidden_dim
        self.latent_dim_vf = value_net_hidden_dim

        # init models
        device = get_device(device)
        
        self.policy_net_class = config['policy_net_class']
        
        if self.image_only or self.policy_net_class == 'MLP':
            self.policy_net = nn.Sequential(
                nn.Linear(input_dim, policy_gn_hidden_dim), activation_fn(),
                nn.Linear(policy_gn_hidden_dim, policy_gn_hidden_dim), activation_fn(),
            ).to(device)
        else:
            graph_network = gn_map[self.policy_net_class]
            self.policy_net = graph_network(
                n_node_features=graph_builder.n_node_feature_dim,
                action_dim=action_dim,
                hidden_dim=policy_gn_hidden_dim,
                n_layers=config['gn_n_layers'],
                propagation_steps=config['gn_propagation_steps'],
                global_avg_pool=config['actor_global_avg_pool'],
                activation_fn=activation_fn
            ).to(device)

        self.value_net = nn.Sequential(
            nn.Linear(input_dim, value_net_hidden_dim), activation_fn(),
            nn.Linear(value_net_hidden_dim, value_net_hidden_dim), activation_fn(),
        ).to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) consisting of
            [batch_size, last_layer_dim_pi] of action latent features, and
            [batch_size, last_layer_dim_vf] of value latent features.
        """
        policy_net_input = features

        if self.policy_net_class != 'MLP':
            policy_net_input = self.graph_builder.graph(features)

        
        return self.policy_net(policy_net_input), self.value_net(features)