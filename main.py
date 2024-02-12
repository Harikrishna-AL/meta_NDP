from meta_guided_ndp import meta_ndp
import networkx as nx
import torch

def main(**wargs):
    config = wargs
    print(config)
    return config['digit1'] + config['digit2']

nums = {'digit1': 1,'digit2': 2}
print(main(**nums))

graph = meta_ndp(nx.Graph(), 10,10,{})
mlp = graph.mlp(10,4,[8,6], torch.nn.Tanh(), False, bias=10)
print(mlp)