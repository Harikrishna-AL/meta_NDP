import networkx as nx
import numpy as np
import torch
from scipy import stats, sparse
from numpy.random import default_rng


class growing_graph:
    def __init__(self, G: nx.Graph, feat_dim, num_nodes, edges):
        self.G = G
        self.feat_dim = feat_dim
        self.num_nodes = num_nodes
        self.edges = edges
        self.graph = nx.Graph()

    def generate_initial_graph(
        self, network_size, sparsity, binary_connectivity, undirected, seed
    ):
        nb_disjoint_initial_graphs = np.inf
        while nb_disjoint_initial_graphs > 1:
            maxWeight = 1
            minWeight = -1
            rng = default_rng(seed)
            if binary_connectivity:
                rvs = stats.uniform(loc=0, scale=1).rvs
                W = np.rint(
                    sparse.random(
                        network_size,
                        network_size,
                        density=sparsity,
                        data_rvs=rvs,
                        random_state=rng,
                    ).toarray()
                )
            else:
                rvs = stats.uniform(loc=minWeight, scale=maxWeight - minWeight).rvs
                W = sparse.random(
                    network_size,
                    network_size,
                    density=sparsity,
                    data_rvs=rvs,
                    random_state=rng,
                ).toarray()  # rows are outbounds, columns are inbounds
            disjoint_initial_graphs = [
                e for e in nx.connected_components(nx.from_numpy_array(W))
            ]
            nb_disjoint_initial_graphs = len(disjoint_initial_graphs)

        if undirected:
            G = nx.from_numpy_array(W, create_using=nx.Graph)
        else:
            G = nx.from_numpy_array(W, create_using=nx.DiGraph)

        return G, W

    def add_new_nodes(self, config, new_nodes_prediction):
        if len(self.G) == 1:
            neighbours = np.array([[0]])
        else:
            neighbours = []
            for idx_node in range(len(self.G)):
                neighbours_idx = [n for n in nx.all_neighbors(self.G, idx_node)]
                neighbours_idx.append(idx_node)
                neighbours.append(np.unique(neighbours_idx))

        # continue to add new nodes based on prediciton

    def mlp(
        self,
        input_dim,
        output_dim,
        hidden_layers_dims,
        activation,
        last_layer_activated,
        bias,
    ):
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_layers_dims[0], bias=bias))
        layers.append(activation)

        for i in range(1, len(hidden_layers_dims)):
            layers.append(
                torch.nn.Linear(
                    hidden_layers_dims[i - 1], hidden_layers_dims[i], bias=bias
                )
            )
            layers.append(activation)

        layers.append(torch.nn.Linear(hidden_layers_dims[-1], output_dim, bias=bias))
        if last_layer_activated:
            layers.append(activation)

        return torch.nn.Sequential(*layers)

    def propagate_features(
        self,
        network_state,
        network_thinking_time,
        activation_function,
        additive_update,
        feature_transformation_model,
        persistent_observation,
    ):
        with torch.no_grad():
            network_state = torch.tensor(network_state, dtype=torch.float32)
            persistent_observation = (
                torch.tensor(persistent_observation, dtype=torch.float32)
                if persistent_observation is not None
                else None
            )

            W = self.G.adjacency_matrix().toarray()
            for step in range(network_thinking_time):
                if additive_update:
                    network_state += W.T @ network_state
                else:
                    network_state = W.T @ network_state

                if feature_transformation_model is not None:
                    network_state = feature_transformation_model(network_state)
                elif activation_function is not None:
                    network_state = activation_function(network_state)

                if persistent_observation is not None:
                    network_state[
                        0 : persistent_observation.shape[0]
                    ] = persistent_observation

        return network_state.detach().numpy()

    def predict_new_nodes(self, config, model):
        pass

    def update_weights(self, config, model):
        pass
