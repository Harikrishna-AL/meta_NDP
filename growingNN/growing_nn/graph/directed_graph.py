import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch_geometric.data import Data
import math
from graph.generated_network import GeneratedNetwork


class DirectedGraph:
    def __init__(self, nodes, edge_dict, num_input_nodes, num_output_nodes, edge_mat):
        self.nodes = nodes.clone().detach().requires_grad_(True)
        self.edge_dict = edge_dict
        self.edge_mat = edge_mat
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes

        node_indices = np.arange(self.nodes.size(0))
        self.input_nodes = list(node_indices[: self.num_input_nodes])
        self.output_nodes = list(
            node_indices[
                self.num_input_nodes : self.num_input_nodes + self.num_output_nodes
            ]
        )

    def add_edges(self, new_edges, new_edge_mat):
        self.edge_mat = new_edge_mat
        for node in new_edges:
            if node not in self.edge_dict:
                self.edge_dict[node] = []
            destinations = new_edges[node]
            for d in destinations:
                if d not in self.edge_dict[node]:
                    self.edge_dict[node].append(d)

        new_mat_size = len(self.nodes)
        new_edge_mat = np.zeros((new_mat_size, new_mat_size))

        for i in range(self.edge_mat.shape[0]):
            for j in range(self.edge_mat.shape[1]):
                new_edge_mat[i, j] = self.edge_mat[i, j]

        for node in new_edges:
            destinations = new_edges[node]
            for d in destinations:
                new_edge_mat[node, d] = 1

        self.edge_mat = np.array(new_edge_mat)

    def add_nodes(self, nodes):
        nodes = nodes.clone().detach().requires_grad_(True)
        self.nodes = torch.vstack((self.nodes, nodes))

    def to_data(self):
        edges = []
        num_nodes = self.nodes.size(0)
        adj_matrix = self.edge_mat

        for node in self.edge_dict:
            destinations = self.edge_dict[node]
            for d in destinations:
                edges.append([node, d])

        edges = torch.tensor(edges).long().t().contiguous().to(self.nodes.device)

        return Data(
            x=self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device),
            edge_index=edges,
            edge_attr=adj_matrix,
        )

    def topologicalSortUtil(self, v, visited, stack):
        visited[v] = True
        if v in self.edge_dict:
            for i in self.edge_dict[v]:
                if not visited[i]:
                    self.topologicalSortUtil(i, visited, stack)
        stack.append(v)

    def topological_sort(self):
        self.V = self.nodes.size(0)
        visited = [False] * self.V
        stack = []
        for i in range(self.V):
            if not visited[i]:
                self.topologicalSortUtil(i, visited, stack)
        return stack[::-1]

    def plot(self, labels=None, fig=None, node_colors=None):
        data = self.to_data()
        G = torch_geometric.utils.to_networkx(data, to_undirected=False)
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
        if fig is None:
            fig = plt.figure()
        canvas = FigureCanvas(fig)

        if node_colors is None:
            node_colors = ["blue"] * self.nodes.size(0)
            for i in self.input_nodes:
                node_colors[i] = "green"
            for i in self.output_nodes:
                node_colors[i] = "red"

        edge_mat = data.edge_attr
        edge_labels = {}
        n = edge_mat.shape[0]
        for i in range(n):
            for j in range(n):
                if edge_mat[i, j] != 0:
                    edge_labels[(i, j)] = str(edge_mat[i, j].item())

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, labels=labels)

        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    def generate_network(self, *args, **kwargs):
        return GeneratedNetwork(self, *args, **kwargs)

    def copy(self):
        nodes = self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device)
        edge_dict = copy.deepcopy(self.edge_dict)
        edge_mat = self.edge_mat
        return DirectedGraph(
            nodes, edge_dict, self.num_input_nodes, self.num_output_nodes, edge_mat
        )
