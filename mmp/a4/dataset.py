from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader


class MMP_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_data: str,
        image_size: int,
        anchor_grid: np.ndarray,
        min_iou: float,
        is_test: bool,
    ):
        """
        @param anchor_grid: The anchor grid to be used for every image
        @param min_iou: The minimum IoU that is required for an overlap for the label grid.
        @param is_test: Whether this is the test set (True) or the validation/training set (False)
        """
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        @return: 3-tuple of image tensor, label grid, and image (file-)number
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()


def get_dataloader(
    path_to_data: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    anchor_grid: np.ndarray,
    is_test: bool,
) -> DataLoader:
    raise NotImplementedError()


def calculate_max_coverage(loader: DataLoader, min_iou: float) -> float:
    """
    @param loader: A DataLoader object, generated with the get_dataloader function.
    @param min_iou: Minimum IoU overlap that is required to count a ground truth box as covered.
    @return: Ratio of how mamy ground truth boxes are covered by a label grid box. Must be a value between 0 and 1.
    """
    raise NotImplementedError()
