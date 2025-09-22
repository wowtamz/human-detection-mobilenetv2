from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# these are the labels from the Cifar10 dataset:
CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class MmpNet(nn.Module):
    """Exercise 2.1"""

    def __init__(self, num_classes: int):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor):
        raise NotImplementedError()


def get_dataloader(
    is_train: bool, data_root: str, batch_size: int, num_workers: int
) -> DataLoader:
    """Exercise 2.2

    @param is_train: Whether this is the training or validation split
    @param data_root: Where to download the dataset to
    @param batch_size: Batch size for the data loader
    @param num_workers: Number of workers for the data loader
    """
    raise NotImplementedError()


def get_criterion_optimizer(model: nn.Module) -> Tuple[nn.Module, optim.Optimizer]:
    """Exercise 2.3a

    @param model: The model that is being trained.
    @return: Returns a tuple of the criterion and the optimizer.
    """
    raise NotImplementedError()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    """Exercise 2.3b

    @param model: The model that should be trained
    @param loader: The DataLoader that contains the training data
    @param criterion: The criterion that is used to calculate the loss for backpropagation
    @param optimizer: Executes the update step
    @param device: The device where the epoch should run on
    """
    raise NotImplementedError()


def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Exercise 2.3c

    @param model: The model that should be evaluated
    @param loader: The DataLoader that contains the evaluation data
    @param device: The device where the epoch should run on

    @return: Returns the accuracy over the full validation dataset as a float."""
    raise NotImplementedError()


def main():
    """Exercise 2.3d"""
    raise NotImplementedError()


if __name__ == "__main__":
    main()
