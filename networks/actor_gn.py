from typing import Optional
import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from stable_baselines3.common import logger
 

class GlobalGraphMeanPool(nn.Module):

    def forward(self, x, batch):
        return global_mean_pool(x, batch)

class ActorGN(nn.Module):

    def __init__(self, 
        action_dim: int,
        n_node_features: int,
        hidden_dim: int,
        activation_fn: nn.Module,
        n_layers: int = 1,
        propagation_steps: int = 1,
        global_avg_pool: bool = True):

        super(ActorGN, self).__init__()

        self.global_avg_pool = global_avg_pool
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.propagation_steps = propagation_steps

        self.activation_fn = activation_fn()
        self.pool = GlobalGraphMeanPool()

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        x = self.node_update(x, edge_index, batch)

        if self.global_avg_pool:
            x = self.pool(x, batch)
        else:
            x = x.reshape(-1, self.action_dim, self.hidden_dim)
        
        return x

    def node_update(self, x, edge_index, batch):
        return x

class NPropGCNConv(GCNConv):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        improved: bool = False, 
        cached: bool = False,
        add_self_loops: bool = True, 
        normalize: bool = True,
        bias: bool = True, 
        propagation_steps: int = 1,
        **kwargs):

        self.propagation_steps = propagation_steps

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            improved=improved,
            cached=cached,
            add_self_loops=add_self_loops,
            normalize=normalize,
            bias=bias,
            **kwargs
        )

    def propagate(self, edge_index, size = None, **kwargs):
        out = None
        for _ in range(self.propagation_steps):
            out = super().propagate(edge_index, size, **kwargs)
        return out

class GCNActor(ActorGN):

    def __init__(self,
        action_dim: int,
        n_node_features: int,
        hidden_dim: int,
        activation_fn: nn.Module,
        n_layers: int = 1,
        propagation_steps: int = 1,
        global_avg_pool: bool = True):

        super().__init__(
            action_dim=action_dim,
            n_node_features=n_node_features,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
            n_layers=n_layers,
            propagation_steps=propagation_steps,
            global_avg_pool=global_avg_pool
        )
        
        self.conv1 = GCNConv(n_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def node_update(self, x, edge_index, batch):
        x = self.activation_fn(self.conv1(x, edge_index))
        x = self.activation_fn(self.conv2(x, edge_index))
        return x

from torch_geometric.nn import GatedGraphConv

class GatedGCActor(ActorGN):

    def __init__(self, 
        action_dim: int,
        n_node_features: int,
        hidden_dim: int,
        activation_fn: nn.Module,
        n_layers: int = 1,
        propagation_steps: int = 1,
        global_avg_pool: bool = True):

        super().__init__(
            action_dim=action_dim,
            n_node_features=n_node_features,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
            n_layers=n_layers,
            propagation_steps=propagation_steps,
            global_avg_pool=global_avg_pool
        )

        if n_node_features > hidden_dim:
            raise ValueError('Input embeddings dim is not allowed to be higher than hidden dim.')

        self.conv = GatedGraphConv(hidden_dim, n_layers)
        
    def node_update(self, x, edge_index, batch):
        return self.activation_fn(self.conv(x, edge_index))

from torch_geometric.nn import TransformerConv

class TransformerActor(ActorGN):

    def __init__(self, 
        action_dim: int,
        n_node_features: int,
        hidden_dim: int,
        activation_fn: nn.Module,
        n_layers: int = 1,
        propagation_steps: int = 1,
        global_avg_pool: bool = True):

        super().__init__(
            action_dim=action_dim,
            n_node_features=n_node_features,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
            n_layers=n_layers,
            propagation_steps=propagation_steps,
            global_avg_pool=global_avg_pool
        )

        self.conv1 = TransformerConv(n_node_features, hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim)
        
    def node_update(self, x, edge_index, batch):
        x = self.activation_fn(self.conv1(x, edge_index))
        x = self.activation_fn(self.conv2(x, edge_index))
        return x

from torch_geometric.nn import GraphConv

class GraphConvActor(ActorGN):

    def __init__(self, 
        action_dim: int,
        n_node_features: int,
        hidden_dim: int,
        activation_fn: nn.Module,
        n_layers: int = 1,
        propagation_steps: int = 1,
        global_avg_pool: bool = True):

        super().__init__(
            action_dim=action_dim,
            n_node_features=n_node_features,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
            n_layers=n_layers,
            propagation_steps=propagation_steps,
            global_avg_pool=global_avg_pool
        )

        self.conv1 = GraphConv(n_node_features, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        
    def node_update(self, x, edge_index, batch):
        x = self.activation_fn(self.conv1(x, edge_index))
        x = self.activation_fn(self.conv2(x, edge_index))
        return x

from torch_geometric.nn import LEConv

class LEConvActor(ActorGN):

    def __init__(self, 
        action_dim: int,
        n_node_features: int,
        hidden_dim: int,
        activation_fn: nn.Module,
        n_layers: int = 1,
        propagation_steps: int = 1,
        global_avg_pool: bool = True):

        super().__init__(
            action_dim=action_dim,
            n_node_features=n_node_features,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
            n_layers=n_layers,
            propagation_steps=propagation_steps,
            global_avg_pool=global_avg_pool
        )

        self.conv1 = LEConv(n_node_features, hidden_dim)
        self.conv2 = LEConv(hidden_dim, hidden_dim)
        
    def node_update(self, x, edge_index, batch):
        x = self.activation_fn(self.conv1(x, edge_index))
        x = self.activation_fn(self.conv2(x, edge_index))
        return x

from torch_geometric.nn import ResGatedGraphConv

class ResGatedGraphConvActor(ActorGN):

    def __init__(self, 
        action_dim: int,
        n_node_features: int,
        hidden_dim: int,
        activation_fn: nn.Module,
        n_layers: int = 1,
        propagation_steps: int = 1,
        global_avg_pool: bool = True):

        super().__init__(
            action_dim=action_dim,
            n_node_features=n_node_features,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
            n_layers=n_layers,
            propagation_steps=propagation_steps,
            global_avg_pool=global_avg_pool
        )

        self.conv1 = ResGatedGraphConv(n_node_features, hidden_dim)
        self.conv2 = ResGatedGraphConv(hidden_dim, hidden_dim)
        
    def node_update(self, x, edge_index, batch):
        x = self.activation_fn(self.conv1(x, edge_index))
        x = self.activation_fn(self.conv2(x, edge_index))
        return x
