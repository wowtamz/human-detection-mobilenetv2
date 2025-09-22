from typing import Tuple
import torch
from torch.utils.data import DataLoader


class MMP_Dataset(torch.utils.data.Dataset):
    """Exercise 3.2"""

    def __init__(self, path_to_data: str, image_size: int):
        """
        @param path_to_data: Path to the folder that contains the images and annotation files, e.g. dataset_mmp/train
        @param image_size: Desired image size that this dataset should return
        """
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        @return: Tuple of image tensor and label. The label is 0 if there is one person and 1 if there a multiple people.
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

def get_dataloader(
        path_to_data: str, image_size: int, batch_size: int, num_workers: int, is_train: bool = True
) -> DataLoader:

    """Exercise 3.2d"""
