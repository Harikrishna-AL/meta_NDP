import numpy as np
import torch
import torch.nn as nn
import torchvision


def x0_sampling(dist, num_parameters):
    if dist == "U[0,1]":
        return np.random.rand(num_parameters)
    elif dist == "N(0,1)":
        return np.random.randn(num_parameters)
    elif dist == "U[-1,1]":
        return 2 * np.random.rand(num_parameters) - 1
    else:
        raise ValueError("Unknown distribution for x0")


def seed_python_numpy_torch_cuda(seed):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_to_patch(image, patch_size):
    image = image.squeeze()
    image = image.permute(1, 2, 0)
    image = image.numpy()
    patches = []
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            patches.append(image[i : i + patch_size, j : j + patch_size])
    return patches


def mnist_data_loader(batch_size=32, shuffle=True):
    mnist_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    mnist_loader = torch.utils.data.DataLoader(
        dataset=mnist_data, batch_size=batch_size, shuffle=shuffle
    )
    return mnist_loader


def get_dims(type):
    if "mnist" in type.lower():
        return 28 * 28, 10


def get_feat(image, **kwargs):
    net = nn.Sequential(
        nn.Conv2d(1, 8, 3, 1),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(8, 16, 3, 1),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(5 * 5 * 16, 64),
    )

    net.parameters = kwargs["cnn_params"]

    return net(image)
