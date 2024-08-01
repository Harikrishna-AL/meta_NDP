from graph.directed_graph import DirectedGraph
from graph.generated_network import GeneratedNetwork
from graph.graph_attention import EAGAttention
from graph.graph_nca import GraphNCA
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from celluloid import Camera
from PIL import Image


def train_epoch(optimizer, nca, graph, loss_list, images, fig, camera, inputs, targets):
    optimizer.zero_grad()

    # Forward pass
    output, nodes = graph.generate_network().batch_forward(inputs)
    print(output.shape, targets.shape)

    # Compute loss
    loss = nn.MSELoss()(output, targets)

    # Backward pass
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    img = graph.plot(fig=fig)
    camera.snap()
    fig.clear()
    images.append(img)
    
    return loss_list, images


def make_initial_graph(config):
    import numpy as np

    NUM_CHANNELS = config["num_channels"]
    input_nodes = config["input_nodes"]
    output_nodes = config["output_nodes"]
    total_nodes = input_nodes + output_nodes
    x = torch.randn((total_nodes, NUM_CHANNELS))
    x[0,1] = 1.0
    x[0,1] = 1.0

    # edge_dict = {0:[2,3], 1:[2,3], 2:[], 3:[]}
    edge_dict = {}
    output_connections = [j for j in range(input_nodes, total_nodes)]
    for i in range(total_nodes):
        if i < input_nodes:
            edge_dict[i] = output_connections
        else:
            edge_dict[i] = []

    edge_mat = torch.zeros((total_nodes, total_nodes))

    for key, values in edge_dict.items():
        for value in values:
            edge_mat[key, value] = np.random.random()

    # print(edge_mat)

    graph = DirectedGraph(nodes= x, edge_dict=edge_dict, num_input_nodes=2, num_output_nodes=2, edge_mat=edge_mat)
    image = graph.plot()
    plt.imshow(image)
    plt.show()

    return graph


def train(config, dataloader):
    graph = make_initial_graph(config)
    hidden_channels = config["hidden_channels"]
    epochs = config["epoch"]
    growth_cycles = config["growth_cycles"]
    nca = GraphNCA(graph, hidden_channels)
    optimizer = torch.optim.Adam(nca.parameters(), lr=0.001)
    loss_list = []
    images = []
    fig = plt.figure()
    camera = Camera(fig)

    #growth loop
    for g in range(growth_cycles):
        graph = nca.grow(graph, 1)

        for _ in range(epochs):
            for inputs, targets in dataloader:
                loss_list, images = train_epoch(optimizer, nca, graph, loss_list, images, fig, camera, inputs, targets)
                print(f"Loss at iteration {_}: {loss_list[-1]}")

        imgs = [Image.fromarray(img) for img in images]
        imgs[0].save("animation_.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)



if __name__ == "__main__":
 
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Take initial graph hyperparameters")
    parser.add_argument("--conf", type=str, default="config.yaml", help="Path to your config yaml with all the hyperparameters.")
    args = parser.parse_args()
    with open(args.conf) as file:
        config = yaml.load(file, Loader = yaml.FullLoader)
    train(config)
