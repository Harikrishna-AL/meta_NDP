from graph.graph_attention import EAGAttention
import random
from typing import Optional
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.nn import GCNConv
import numpy as np


class GraphNCA(nn.Module):
    def __init__(self, graph, num_hidden_channels: int = 16, max_replications: int = 2):
        super().__init__()
        self.graph = graph
        self.num_input_nodes = self.graph.num_input_nodes
        self.num_output_nodes = self.graph.num_output_nodes
        self.input_nodes = self.graph.input_nodes
        self.output_nodes = self.graph.output_nodes
        self.value_idx = 0
        self.replication_idx = 1
        self.operations = [torch.add, torch.subtract, torch.multiply]
        self.activations = [torch.relu, torch.tanh]
        self.replicated_cells = []
        self.num_operations = len(self.operations)
        self.num_activations = len(self.activations)
        self.operation_channels = [2, 4]
        self.activation_channels = [5, 6]
        self.num_hidden_channels = num_hidden_channels
        self.num_channels = self.get_number_of_channels(
            self.num_operations, self.num_activations, self.num_hidden_channels
        )
        self.num_nodes = self.graph.to_data().x.shape[0]
        self.perception_net = GCNConv(
            self.num_channels, self.num_channels * 3, bias=False
        )
        self.global_atten = EAGAttention(
            num_nodes=self.num_nodes, num_channels=self.num_channels
        )
        self.update_net = nn.Sequential(
            nn.Linear(self.num_channels, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_channels),
        )
        self.replication_network = nn.Linear(self.num_channels, self.num_channels)
        self.max_replications = max_replications

    @classmethod
    def get_number_of_channels(
        cls, num_operations: int, num_activations: int, num_hidden_channels
    ):
        return num_operations + num_activations + num_hidden_channels + 2

    def forward(self, data):
        nodes = data.x.clone().detach().requires_grad_(True)
        if isinstance(data.edge_attr, np.ndarray):
            edges = (
                torch.tensor(data.edge_attr, dtype=torch.float32)
                .clone()
                .detach()
                .cpu()
                .numpy()
            )
        else:
            edges = data.edge_attr.clone().detach().cpu().numpy()

        if nodes.dim() != 3:
            nodes = nodes.unsqueeze(0)

        features, edge_features = self.global_atten(nodes, edges)

        update = self.update_net(features)
        nodes = nodes + update

        # Convert edge_features to numpy array
        new_edge_mat = edge_features.squeeze(0).clone().detach().cpu().numpy()

        return nodes.squeeze(0), new_edge_mat

    def replicate(self, x, edge_dict):
        num_nodes = x.shape[0]
        current_count = num_nodes
        dist = Bernoulli(logits=x[:, self.replication_idx])
        ready_to_replicate = dist.sample().squeeze()
        ready_to_replicate_indices = [
            i
            for i in range(len(ready_to_replicate))
            if ready_to_replicate[i] == 1.0 and i not in self.output_nodes
        ]
        ready_to_replicate_indices = ready_to_replicate_indices[: self.max_replications]
        if len(ready_to_replicate_indices) > 0:
            children = self.replication_network(x[ready_to_replicate_indices])
            new_edge_dict = {}
            for parent_node in ready_to_replicate_indices:
                parent_destinations = edge_dict[parent_node]
                random_destination = random.choices(parent_destinations, k=1)
                new_edge_dict[current_count] = random_destination
                new_edge_dict[parent_node] = [current_count]
                current_count += 1
            return children, new_edge_dict
        return None, None

    def grow(
        self, graph, num_iterations: int = 1, replicate_interval: Optional[int] = 1
    ):
        new_graph = graph.copy()
        for i in range(num_iterations):
            data = new_graph.to_data()
            x, new_edge_mat = self.forward(data)
            if replicate_interval is not None:
                if i % replicate_interval == 0:
                    children, new_edge_dict = self.replicate(
                        x,
                        new_graph.edge_dict,
                    )
                    if children is not None:
                        new_graph.add_nodes(children)
                        new_graph.add_edges(new_edge_dict, new_edge_mat)
        return new_graph
