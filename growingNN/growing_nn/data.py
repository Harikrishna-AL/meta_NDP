import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
from tqdm import tqdm


class ARC(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []

        for file in os.listdir(data_dir):
            if file.endswith(".json"):
                with open(os.path.join(data_dir, file), "r") as f:
                    task = json.load(f)
                    if train:
                        self.data.extend(task["train"])
                    else:
                        self.data.extend(task["test"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grid = self.data[idx]
        input_grid = torch.tensor(grid["input"], dtype=torch.float32)
        output_grid = torch.tensor(grid["output"], dtype=torch.float32)

        sample = {"input_grid": input_grid, "output_grid": output_grid}

        if self.transform:
            sample = self.transform(sample)

        return sample


def pad_grid(batch):
    input_grids = [item["input_grid"] for item in batch]
    output_grids = [item["output_grid"] for item in batch]

    max_height = max([grid.shape[0] for grid in input_grids])
    max_width = max([grid.shape[1] for grid in input_grids])

    padded_input_grids = [
        torch.nn.functional.pad(
            grid, (0, max_width - grid.shape[1], 0, max_height - grid.shape[0])
        )
        for grid in input_grids
    ]
    padded_output_grids = [
        torch.nn.functional.pad(
            grid, (0, max_width - grid.shape[1], 0, max_height - grid.shape[0])
        )
        for grid in output_grids
    ]

    return {
        "input_grids": torch.stack(padded_input_grids),
        "output_grids": torch.stack(padded_output_grids),
    }


def get_data_loader(data_dir, batch_size, train=True):
    dataset = ARC(data_dir, train=train)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_grid
    )
    return data_loader


# create MNIST dataset and dataloader


def get_mnist_data_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, test_loader


# if __name__ == "__main__":
#     data_dir = "/Users/vishal/Desktop/NDP_Research/ARC-AGI/data/training"
#     data_loader = get_data_loader(data_dir, 16)
#     for i, sample in enumerate(data_loader):
#         print(f"Batch {i}")
#         print(sample)
#         # break
# if __name__ == "__main__":
#     train_loader, test_loader = get_mnist_data_loader(16)
#     for i, sample in tqdm(train_loader):
#         i.squeeze(1)
#         i = i.view(i.size(0), -1)
#         print(i.shape)
#         print(sample.shape)
#         break
