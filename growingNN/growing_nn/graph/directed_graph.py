import copy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch_geometric.data import Data
import math

from growing_nn.graph.generated_network import GeneratedNetwork


class DirectedGraph:
    def __init__(self, nodes, edge_dict, num_input_nodes, num_output_nodes, edge_mat):
        self.nodes = nodes
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
        # add edges to edge_mat
        # for node in new_edges:
        #     destinations = new_edges[node]
        #     for d in destinations:
        #         self.edge_mat[node, d] = 1
        # print(self.edge_mat)
        new_mat_size = len(self.nodes)
        new_edge_mat = np.zeros((new_mat_size, new_mat_size))   
        # add older edges to new edge matrix
        # print(self.edge_mat.shape)
        # print(self.edge_mat)
        for i in range(self.edge_mat.shape[0]):
            for j in range(self.edge_mat.shape[1]):
                # print(self.edge_mat[i, j])
                new_edge_mat[i, j] = self.edge_mat[i, j]
                
        for node in new_edges:
            destinations = new_edges[node]
            for d in destinations:
                new_edge_mat[node, d] = 1
            
        # #return as numpy
        # print(self.edge_mat)
        # print(new_edges)
        self.edge_mat = np.array(new_edge_mat)

    def add_nodes(self, nodes):
        self.nodes = torch.vstack((self.nodes, nodes))

    def to_data(self):
        edges = []
        num_nodes = self.nodes.size(0)
        # adj_matrix = torch.zeros((num_nodes, num_nodes), device=self.nodes.device)
        adj_matrix = self.edge_mat

        for node in self.edge_dict:
            destinations = self.edge_dict[node]
            for d in destinations:
                edges.append([node, d])
                # adj_matrix[node, d] = 1

        edges = torch.tensor(edges).long().t().contiguous().to(self.nodes.device)

        return Data(
            x=self.nodes * torch.ones(self.nodes.size(), device=self.nodes.device),
            edge_index=edges,
            edge_attr=adj_matrix,
        )

    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        if v in self.edge_dict:
            for i in self.edge_dict[v]:
                if not visited[i]:
                    self.topologicalSortUtil(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.append(v)

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topological_sort(self):
        # Mark all the vertices as not visited
        self.V = self.nodes.size(0)
        visited = [False] * self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if not visited[i]:
                self.topologicalSortUtil(i, visited, stack)

        # Print contents of the stack
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

        #plot the edge values too using edge_mat
        edge_mat = data.edge_attr
        # print(edge_mat)
        edge_labels = {}
        n = int(math.sqrt(edge_mat.size))
        for i in range(n):
            for j in range(n):
                if edge_mat[i, j] != 0:
                    edge_labels[(i, j)] = str(edge_mat[i, j].item())


        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, labels=labels)

        canvas.draw()  # draw the canvas, cache the renderer

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
