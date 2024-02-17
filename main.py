from meta_guided_ndp import meta_ndp
import networkx as nx
import torch
import matplotlib.pyplot as plt


def main(**wargs):
    config = wargs
    print(config)
    return config["digit1"] + config["digit2"]


nums = {"digit1": 1, "digit2": 2}
print(main(**nums))

graph = meta_ndp(nx.Graph(), 10, 10, {})
mlp = graph.mlp(10, 4, [8, 6], torch.nn.Tanh(), False, bias=10)
print(mlp)
init_graph, w = graph.generate_initial_graph(
    network_size=4, sparsity=0.5, seed=0, binary_connectivity=True, undirected=True
)
# visualize the graph and edge weights
pos = nx.spring_layout(init_graph)
nx.draw(init_graph, pos, with_labels=True, font_weight="bold")
print(w)
# save the graph as a png file
plt.savefig("graph.png")

# state of size  4*8
import numpy as np

net_state = np.random.rand(4, 8)
pair_dict = graph.pair_embeddings(W=w, network_state=net_state, self_link=True)
print(pair_dict)
