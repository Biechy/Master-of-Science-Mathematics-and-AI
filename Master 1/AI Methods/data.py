"""
Module to load datasets CIFAR-10 : https://www.cs.toronto.edu/~kriz/cifar.html
"""
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, Subset
from torchvision import datasets, transforms


def LoadCIFAR10(batch_size):
    """
    Load CIFAR10 dataset
    Parameters
    -----------
    batch-size : int
                 batch-size for mini batch wanted
    Returns
    -----------
    Dataloaders
        train dataset and test dataset
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    training_data = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )

    test_data = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )

    dwn_images(training_data)

    train = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train, test


def LoadCIFAR10_subset(batch_size, subset):
    """
    Load CIFAR10 dataset
    Parameters
    -----------
    batch-size : int
                 batch-size for mini batch wanted

    subset : int
                size of the subset
    Returns 
    -----------
    Dataloaders
        subset of train dataset and test dataset
    """

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    subset_indices = range(5000)   # indices of the subset
    train_subset = Subset(training_data, subset_indices)

    # create a dataloader for the subset
    train = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train, test


def dwn_images(training_data):
    """
    Downlaod an image of 25 exemples of the dataset
    Parameters
    ----------
    training_data : DataLoader
    """
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.numpy().swapaxes(0, 1).swapaxes(1, 2))
    plt.savefig("./results/imgs.png")
