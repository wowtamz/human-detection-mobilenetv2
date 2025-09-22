import torch
import torch.optim as optim

from .model import MmpNet


def step(
    model: MmpNet,
    criterion,
    optimizer: optim.Optimizer,
    img_batch: torch.Tensor,
    lbl_batch: torch.Tensor,
) -> float:
    """Performs one update step for the model

    @return: The loss for the specified batch. Return a float and not a PyTorch tensor
    """
    raise NotImplementedError()


def get_random_sampling_mask(labels: torch.Tensor, neg_ratio: float) -> torch.Tensor:
    """
    @param labels: The label tensor that is returned by your data loader.
    The values are either 0 (negative label) or 1 (positive label).
    @param neg_ratio: The desired negative/positive ratio.
    Hint: after computing the mask, check if the neg_ratio is fulfilled.
    @return: A tensor with the same shape as labels
    """
    assert labels.min() >= 0 and labels.max() <= 1  # remove this line if you want
    raise NotImplementedError()


def main():
    """Put your training code for exercises 5.2 and 5.3 here"""
    raise NotImplementedError()


if __name__ == "__main__":
    main()
