from graph.directed_graph import DirectedGraph
from graph.generated_network import GeneratedNetwork
from graph.graph_attention import EAGAttention
from graph.graph_nca import GraphNCA

import torch


def train_epoch():
    

def make_initial_graph():
    pass

def train():
    pass

if __name__ == "__main__":
 
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Take initial graph hyperparameters")
    parser.add_argument("--conf", type=str, default="config.yaml", help="Path to your config yaml with all the hyperparameters.")
    args = parser.parse_args()
    with open(args.conf) as file:
        config = yaml.load(file, Loader = yaml.FullLoader)

