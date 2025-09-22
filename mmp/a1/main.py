from typing import Sequence
import torch


def build_batch(paths: Sequence[str], transform=None) -> torch.Tensor:
    """Exercise 1.1

    @param paths: A sequence (e.g. list) of strings, each specifying the location of an image file.
    @param transform: One or multiple image transformations for augmenting the batch images.
    @return: Returns one single tensor that contains every image.
    """
    raise NotImplementedError()


def get_model() -> torch.nn.Module:
    """Exercise 1.2

    @return: Returns a neural network, initialised with pretrained weights.
    """
    raise NotImplementedError()


def main():
    """Exercise 1.3

    Put all your code for exercise 1.3 here.
    """
    raise NotImplementedError()


if __name__ == "__main__":
    main()
