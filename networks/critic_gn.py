import torch
from torch import nn
from torch_geometric.nn import GraphConv, global_mean_pool
from stable_baselines3.common import logger

class CriticGN(nn.Module):

    def __init__(self, 
        n_node_features=2, 
        hidden_dim: int = 64,
        output_dim: int = 64,
        global_aggregation=True,
        activation_fn=nn.Tanh):

        super(CriticGN, self).__init__()

        self.output_dim = output_dim
        self.global_aggregation = global_aggregation
        self.activation_fn = activation_fn()

        # graph convolution layers
        self.conv1 = GraphConv(n_node_features, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        # assume (batch_size, n_nodes * n_features)
        assert len(x.shape) == 2

        # graph convolutions
        x1 = self.activation_fn(self.conv1(x, edge_index))
        x2 = self.activation_fn(self.conv2(x1, edge_index))

        if self.global_aggregation:
          xout = global_mean_pool(x2, batch)
        
        # logging
        logger.record_mean('critic/0/input', torch.mean(x, dim=0))
        logger.record_mean('critic/1/activations/conv1', torch.mean(x1, dim=0))
        logger.record_mean('critic/2/activations/conv2', torch.mean(x2, dim=0))
        logger.record_mean('critic/3/output', torch.mean(xout, dim=0))

        return xout
