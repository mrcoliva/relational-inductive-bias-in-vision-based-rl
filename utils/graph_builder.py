import torch
from torch_geometric.data import Data, Batch

class GraphBuilder(object):

  def __init__(
    self,
    n_nodes: int,
    n_shared_features: int,
    n_features_per_node: int,
    add_forward_skip_connections: bool = False,
    add_end_effector_connections: bool = False,
    add_self_edges: bool = True):
    
    self.n_nodes = n_nodes
    self.n_shared_features = n_shared_features
    self.n_features_per_node = n_features_per_node
    self.add_forward_skip_connections = add_forward_skip_connections
    self.add_end_effector_connections = add_end_effector_connections
    self.add_self_edges = add_self_edges

    # final dimension of each node's features vector
    self.n_node_feature_dim = self.n_shared_features + self.n_features_per_node

    # expected dimension of flat input vector per sample
    self.sample_dim = self.n_shared_features + self.n_nodes * self.n_features_per_node

    # precompute constant graph adjacency
    self.edge_index = self._adjacency(with_self_edges=add_self_edges)
    self.edges_cache = dict()

  def node_embeddings(self, observations):
    """
    Converts flattened observation vector into node embedding vectors
    by combining the shared part of the features with node level features.
    (batch_size, sample_dim) -> (batch_sze * n_nodes, n_node_features_dim) 
    """
    assert len(observations.shape) == 2
    assert observations.shape[1] == self.sample_dim,\
      f'{observations.shape[1]} != {self.sample_dim}'

    batch_size = observations.shape[0]
    
    x_shared = observations[:, :self.n_shared_features]\
      .repeat_interleave(self.n_nodes, dim=0)
    x_nodes = observations[:, self.n_shared_features:]\
      .reshape(batch_size * self.n_nodes, self.n_features_per_node)

    embeddings = torch.cat([x_shared, x_nodes], dim=1)

    # verify embeddings shape
    expected_shape = torch.Size([batch_size * self.n_nodes, self.n_node_feature_dim])
    assert embeddings.shape == expected_shape,\
      f'{embeddings.shape} != {expected_shape}'

    return embeddings

  def _adjacency(self, with_self_edges: bool = True):
    """
    Computes the adjacency matrix of a kinematic chain.
    """
    edges = []

    for i in range(0, self.n_nodes):
      if i > 0:
        edges.append([i, i-1])

      if with_self_edges:
        edges.append([i, i])
      
      # edges for kinematic chain
      if i < self.n_nodes - 1:
        edges.append([i, i + 1])

        if self.add_forward_skip_connections and self.n_nodes > (i + 2):
          for j in range(i + 2, self.n_nodes):
            edges.append([i, j])

        if self.add_end_effector_connections and i < (self.n_nodes - 2):
            edges.append([self.n_nodes - 1, i])

    edge_index = torch.tensor(edges).t().contiguous()
    return edge_index

  def graph(self, observations):
    if not isinstance(observations, torch.Tensor):
      x = torch.tensor([observations])
    else:
      x = observations

    embeddings = self.node_embeddings(x)
    graph = self.build_graph(embeddings)

    return graph

  def build_graph(self, x):
    if x.shape[0] == 1:
      return Data(x=x, edge_index=self.edge_index)

    elif x.shape[0] > 1:
      batch_size = int(x.shape[0] / self.n_nodes)

      try:
        edge_index, node_to_graph_map = self.edges_cache[batch_size]
      except KeyError:
        # create graph batch: one large graph with disconnected subgraphs
        all_edges = self.edge_index.repeat((1, batch_size))
        mask = (torch.arange(batch_size) *
                self.n_nodes).repeat_interleave(self.edge_index.shape[1]).repeat((2, 1))

        edge_index = all_edges + mask
        node_to_graph_map = torch.arange(batch_size).repeat_interleave(self.n_nodes)
        self.edges_cache[batch_size] = (edge_index, node_to_graph_map)
      
      return Batch(batch=node_to_graph_map, x=x, edge_index=edge_index)

    else:
      raise ValueError(f'Unexpected observation shape: {x.shape}')

class NerveNetGraphBuilder(GraphBuilder):

  def __init__(
    self,
    n_nodes: int,
    n_shared_features: int,
    n_features_per_node: int,
    add_forward_skip_connections: bool = False,
    add_self_edges: bool = True):

    super(NerveNetGraphBuilder, self).__init__(
      n_nodes=n_nodes + 1, # add one global node
      n_shared_features=n_shared_features,
      n_features_per_node=n_features_per_node,
      add_forward_skip_connections=add_forward_skip_connections,
      add_self_edges=add_self_edges
    )

    # final dimension of each node's features vector
    self.n_node_feature_dim = self.n_features_per_node

    # expected dimension of flat input vector per sample
    # each node (incl. the global node) has the same feature dim
    self.sample_dim = self.n_nodes * self.n_features_per_node

  def _adjacency(self, with_self_edges: bool = True):
    """
    Computes the adjacency matrix of a kinematic chain.
    """
    edges = []

    for i in range(0, self.n_nodes - 1):
      if i > 0:
        edges.append([i, i-1])

      if with_self_edges:
        edges.append([i, i])
      
      # edges for kinematic chain
      if i < self.n_nodes - 2:
        edges.append([i, i + 1])

        if self.add_forward_skip_connections and self.n_nodes > (i + 2):
          for j in range(i + 2, self.n_nodes - 1):
            edges.append([i, j])

    # connect to global node
    for i in range(0, self.n_nodes - 1):
      edges.append([i, self.n_nodes - 1])
      edges.append([self.n_nodes - 1, i])

    edge_index = torch.tensor(edges).t().contiguous()
    return edge_index

  def node_embeddings(self, observations):
    """
    Converts flattened observation vector into node embedding vectors
    by combining the shared part of the features with node level features.
    (batch_size, sample_dim) -> (batch_sze * n_nodes, n_node_features_dim) 
    """
    assert len(observations.shape) == 2
    # assert observations.shape[1] == self.sample_dim,\
    #   f'{observations.shape[1]} != {self.sample_dim}'

    batch_size = observations.shape[0]
    
    x_shared = observations[:, :self.n_shared_features]
    
    x_nodes = observations[:, self.n_shared_features:]\
      .reshape(batch_size * (self.n_nodes - 1), self.n_features_per_node)

    embeddings = torch.cat([x_shared, x_nodes], dim=0)

    # verify embeddings shape
    expected_shape = torch.Size([batch_size * self.n_nodes, self.n_features_per_node])
    assert embeddings.shape == expected_shape,\
      f'{embeddings.shape} != {expected_shape}'

    return embeddings