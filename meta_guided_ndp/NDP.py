import networkx as nx
import numpy as np
import torch

class growing_graph():

    def __init__(self, G: nx.Graph, feat_dim, num_nodes, edges):
        self.G = G
        self.feat_dim = feat_dim
        self.num_nodes = num_nodes
        self.edges = edges
        self.graph = nx.Graph()

    def add_new_nodes(self, config, new_nodes_prediction):
        if len(self.G) == 1:
            neighbours = np.array([[0]])
        else:
            neighbours = []
            for idx_node in range(len(G)):
                neighbours_idx = [n for n in nx.all_neighbors(G, idx_node)]
                neighbours_idx.append(idx_node)
                neighbours.append(np.unique(neighbours_idx))

        # continue to add new nodes based on prediciton

    def mlp(self,
            input_dim,
            output_dim,
            hidden_layers_dims,
            activation,
            last_layer_activated,
            bias):
        
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_layers_dims[0], bias=bias))
        layers.append(activation)

        for i in range(1, len(hidden_layers_dims)):
            layers.append(torch.nn.Linear(hidden_layers_dims[i-1], hidden_layers_dims[i], bias=bias))
            layers.append(activation)

        layers.append(torch.nn.Linear(hidden_layers_dims[-1],output_dim, bias=bias))
        if last_layer_activated:
            layers.append(activation)
        
        return torch.nn.Sequential(*layers)
        

    def propagate_features(
            self, 
            network_thinking_time, 
            activation_function, 
            additive_update, 
            feature_transformation_model):
        pass

    def predict_new_nodes(self,config, model):
        pass

    def update_weights(self,config, model):
        pass