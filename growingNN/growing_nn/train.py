from graph.directed_graph import DirectedGraph
from graph.generated_network import GeneratedNetwork
from graph.graph_attention import EAGAttention
from graph.graph_nca import GraphNCA
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from celluloid import Camera
from PIL import Image
from data import get_mnist_data_loader
from tqdm import tqdm


def train_epoch(optimizer, nca, graph, loss_list, images, fig, camera, inputs, targets):
    optimizer.zero_grad()

    # Forward pass
    output, nodes = graph.generate_network().batch_forward(inputs)
    # print(output.shape, targets.shape)

    # Compute loss
    # print(output.shape, targets.shape)
    output = nn.Softmax(dim=1)(output)
    # take argmax accross all batches
    # output = torch.argmax(output, dim=1)
    # print(output.shape, targets.shape)
    # print(output, targets)
    loss = nn.CrossEntropyLoss()(output, targets)

    # Backward pass
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    # img = graph.plot(fig=fig)
    # camera.snap()
    # fig.clear()
    # images.append(img)

    return loss_list, images


def make_initial_graph(config):
    import numpy as np

    NUM_OPERATIONS = config["num_operations"]
    NUM_ACTIVATIONS = config["num_activations"]
    NUM_HIDDEN_CHANNELS = config["hidden_channels"]
    NUM_CHANNELS = GraphNCA.get_number_of_channels(
        NUM_OPERATIONS, NUM_ACTIVATIONS, NUM_HIDDEN_CHANNELS
    )

    input_nodes = config["input_nodes"]
    output_nodes = config["output_nodes"]
    total_nodes = input_nodes + output_nodes
    x = torch.randn((total_nodes, NUM_CHANNELS))
    x[0, 1] = 1.0
    x[0, 1] = 1.0

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

    # graph = DirectedGraph(nodes= x, edge_dict=edge_dict, num_input_nodes=2, num_output_nodes=2, edge_mat=edge_mat)
    graph = DirectedGraph(
        nodes=x,
        edge_dict=edge_dict,
        num_input_nodes=input_nodes,
        num_output_nodes=output_nodes,
        edge_mat=edge_mat,
    )
    # image = graph.plot()
    # plt.imshow(image)
    # plt.savefig("initial_graph.png")
    # plt.show()

    # grow the graph till there are enough nodes
    
    return graph


def evaluate(nca, graph, dataloader):
    val_loss = []
    for inputs, targets in tqdm(dataloader):
        inputs = inputs.squeeze(1)
        inputs = inputs.view(inputs.size(0), -1)
        output, nodes = graph.generate_network().batch_forward(inputs)
        loss = nn.MSELoss()(output, targets)
        print(f"Validation Loss: {loss.item()}")
        val_loss.append(loss.item())

    return val_loss


def train(config, dataloader, eval_dataloader):
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
    input_nodes = config["input_nodes"]

    # growth loop
    print("Growing the graph to make it large enough...")
    for z in range(input_nodes):
        graph = nca.grow(graph, 1)
    print("Graph grown to desired size!")

    print("Training the model...")
    for g in range(growth_cycles):
        graph = nca.grow(graph, 1)

        for _ in range(epochs):
            for inputs, targets in tqdm(dataloader):
                # convert input image to single linear array
                inputs = inputs.squeeze(1)
                inputs = inputs.view(inputs.size(0), -1)
                loss_list, images = train_epoch(
                    optimizer,
                    nca,
                    graph,
                    loss_list,
                    images,
                    fig,
                    camera,
                    inputs,
                    targets,
                )
            print(f"Loss at iteration {_}: {loss_list[-1]}")

        # evaluate the model
        val_loss = evaluate(nca, graph, eval_dataloader)
        # plot validation loss and training loss and save the plot
        plt.plot(val_loss, label="Validation Loss")
        plt.plot(loss_list, label="Training Loss")
        plt.legend()
        plt.savefig(f"loss_plot_{g}.png")

    # imgs = [Image.fromarray(img) for img in images]
    # imgs[0].save(
    #     "animation_.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0
    # )


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Take initial graph hyperparameters")
    parser.add_argument(
        "--conf",
        type=str,
        default="config.yaml",
        help="Path to your config yaml with all the hyperparameters.",
    )
    args = parser.parse_args()
    with open(args.conf) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    dataloader, eval_dataloader = get_mnist_data_loader(config["batch_size"])
    train(config, dataloader, eval_dataloader)
