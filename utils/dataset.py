from typing import Tuple
import numpy as np
import torch
import torchvision
import os
from torch.utils.data import DataLoader
from PIL import Image

from . import annotation
from utils import label_grid
from utils.transforms import get_transforms, apply_transforms_to_annotations, apply_transforms_to_img


class MMP_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_data: str,
        image_size: int,
        anchor_grid: np.ndarray,
        min_iou: float,
        is_test: bool,
        augmentations: list = []
    ):
        """
        @param anchor_grid: The anchor grid to be used for every image
        @param min_iou: The minimum IoU that is required for an overlap for the label grid.
        @param is_test: Whether this is the test set (True) or the validation/training set (False)
        """
        self.path_to_data = path_to_data if path_to_data.endswith("/") else path_to_data + "/"
        self.image_size = image_size
        self.anchor_grid = anchor_grid
        self.min_iou = min_iou
        self.is_test = is_test

        path = list(f"{path_to_data}/{p}" for p in os.listdir(path_to_data) if p.endswith(".jpg"))
        self.image_paths = list(sorted(path))
        self.annotation_dict = self.get_annotation_dict()
        self.augmentations = augmentations

    def get_annotation_dict(self) -> dict:
        dictionary = dict()
        for i in self.image_paths:
            key = i.removeprefix(self.path_to_data).removesuffix(".jpg")
            gt_path = i.replace(".jpg", ".gt_data.txt")
            if os.path.exists(gt_path):
                dictionary[key] = annotation.read_groundtruth_file(gt_path)
            else:
                dictionary[key] = []
        return dictionary
    
    def get_img_scale(self, img_id) -> float:
        img_path = self.get_image_path(img_id)
        img = Image.open(img_path)
        width, height = img.size
        scale = self.image_size / max(width, height)
        return scale
    
    def get_rescaled_annotation(self, img_id, rect: annotation.AnnotationRect):
        scale = self.get_img_scale(img_id)
        return rect.rescaled(scale)

    def get_image_id(self, image_path):
        return image_path.removeprefix(self.path_to_data).removesuffix(".jpg")
    
    def get_image_path(self, image_id):
        return self.path_to_data + f"{image_id}.jpg"
    
    def get_annotations(self, img_id) -> list:
        return self.annotation_dict[img_id]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        @return: 3-tuple of image tensor, label grid, and image (file-)number
        """
        img_path = self.image_paths[idx]
        img_id = self.get_image_id(img_path)
        img = Image.open(img_path)

        scale, padding = get_scale_and_padding(img, self.image_size)

        transforms = get_transforms(self.image_size, padding, self.augmentations)
        img_tensor = apply_transforms_to_img(img, transforms)

        annotations = self.get_annotations(img_id)
        annotations = apply_transforms_to_annotations(annotations, scale, transforms, self.image_size)

        l_grid = torch.Tensor() if len(annotations) == 0 else label_grid.get_label_grid(self.anchor_grid, annotations, self.min_iou)

        return (img_tensor, l_grid, img_id)

    def __len__(self) -> int:
        return len(self.image_paths)

def get_scale_and_padding(img, size):
    width, height = img.size
    size_delta = abs(height - width)
    pad_x = size_delta if height > width else 0
    pad_y = size_delta if height < width else 0
    padding = (0, 0, pad_x, pad_y)
    scale = size / max(width, height)
    return scale, padding

def get_dataloader(
    path_to_data: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    anchor_grid: np.ndarray,
    min_iou: float,
    is_test: bool,
    augmentations: list = []
) -> DataLoader:
    
    dataset = MMP_Dataset(path_to_data, image_size, anchor_grid, min_iou, is_test=is_test, augmentations=augmentations)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=not is_test,
                            drop_last=not is_test)
    
    return dataloader