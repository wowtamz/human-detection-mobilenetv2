from typing import List, Tuple
import torch
import numpy as np

from ..a5.model import MmpNet
from ..a3.annotation import AnnotationRect


def batch_inference(
    model: MmpNet, images: torch.Tensor, device: torch.device, anchor_grid: np.ndarray
) -> List[List[Tuple[AnnotationRect, float]]]:
    raise NotImplementedError()


def evaluate() -> float:  # feel free to change the arguments
    """Evaluates a specified model on the whole validation dataset.

    @return: AP for the validation set as a float.

    You decide which arguments this function should receive
    """
    raise NotImplementedError()


def evaluate_test():  # feel free to change the arguments
    """Generates predictions on the provided test dataset.
    This function saves the predictions to a text file.

    You decide which arguments this function should receive
    """
    raise NotImplementedError()


def main():
    """Put the surrounding training code here. The code will probably look very similar to last assignment"""
    raise NotImplementedError()


if __name__ == "__main__":
    main()
