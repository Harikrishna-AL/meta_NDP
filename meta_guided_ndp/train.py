from NDP import growing_graph as meta_ndp
from utils import seed_python_numpy_torch_cuda

import numpy as np
import torch

def fitness_functional(config: dict, graph: meta_ndp):
    def fitness(evolved_parameters: np.array):
        mean_reward = 0

        for _ in range(config['num_growth_evals']):
            if config['num_growth_evals'] > 1:
                config['seed'] = None

            seed_python_numpy_torch_cuda(config['seed'])

            # init graph
            G = graph.generate_initial_graph(
                network_size=config['network_size'],
                sparsity=config['sparsity'],
                binary_connectivity=config['binary_connectivity'],
                undirected=config['undirected'],
                seed=config['seed'],
                )
            
            # init network state
            if config['coevolve_initial_embd']:
                initial_network_state = np.expand_dims(evolved_parameters[:, config['node_embedding_size']], axis=0)
            elif config['shared_initial_embd'] and config['random_initial_embd']:
                initial_network_state = config['initial_network_state']
            else:
                initial_network_state = np.random.rand(config['initial_network_size'], config['node_embedding_size'])

            # create growth decision network
            mlp_growth_model = graph.mlp(
                input_dim=config['node_size _growth_model'],
                output_dim=1,
                hidden_layers_dims=config['mlp_growth_hidden_layers_dims'],
                last_layer_activated=config["growth_model_last_layer_activated"],
                activation=torch.nn.Tanh(),
                bias=config['growth_model_bias'],
            )
            
                


    return fitness


def main():
    pass
