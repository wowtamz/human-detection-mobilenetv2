from typing import Tuple
import numpy as np
import torch
import torchvision
import os
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from pathlib import Path
from ..a3 import annotation
from . import anchor_grid
from . import label_grid


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
        self.path_to_data = path_to_data
        self.image_size = image_size
        self.anchor_grid = anchor_grid
        self.min_iou = min_iou
        self.is_test = is_test

        path = list(f"{path_to_data}/{p}" for p in os.listdir(path_to_data) if p.endswith(".jpg"))
        self.image_paths = list(sorted(path))
        self.annotation_dict = self.get_annotation_dict()

    def get_annotation_dict(self) -> dict:
        dictionary = dict()
        for i in self.image_paths:
            gt_path = i.replace(".jpg", ".gt_data.txt")
            if os.path.exists(gt_path):
                dictionary[i] = annotation.read_groundtruth_file(gt_path)
        return dictionary


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        @return: 3-tuple of image tensor, label grid, and image (file-)number
        """
        img_path = self.image_paths[idx]
        img_id = img_path.removeprefix(self.path_to_data + "/").removesuffix(".jpg")
        #img_id = img_id.removesuffix(".jpg")
        img = Image.open(img_path)
        width, height = img.size
        size_delta = abs(height - width)
        pad_right = size_delta if height > width else 0
        pad_bottom = size_delta if height < width else 0
        padding = (0, 0, pad_right, pad_bottom)
        
        tfm = torchvision.transforms.Compose([
            torchvision.transforms.Pad(padding, 0, "constant"),
            torchvision.transforms.Resize((self.image_size, self.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_tensor = tfm(img)
        l_grid = torch.Tensor() if self.is_test else label_grid.get_label_grid(self.anchor_grid, self.annotation_dict[img_path], self.min_iou)

        return (img_tensor, l_grid, img_id)

    def __len__(self) -> int:
        return len(self.image_paths)

def get_dataloader(
    path_to_data: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    anchor_grid: np.ndarray,
    is_test: bool,
) -> DataLoader:
    
    dataset = MMP_Dataset(path_to_data, image_size, anchor_grid, 0.7, is_test=is_test)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=not is_test,
                            drop_last=not is_test)
    
    return dataloader


def calculate_max_coverage(loader: DataLoader, min_iou: float) -> float:
    """
    @param loader: A DataLoader object, generated with the get_dataloader function.
    @param min_iou: Minimum IoU overlap that is required to count a ground truth box as covered.
    @return: Ratio of how mamy ground truth boxes are covered by a label grid box. Must be a value between 0 and 1.
    """
    raise NotImplementedError()

if __name__ == "__main__":
    train_data_dir = f"{Path.cwd()}/dataset/train"

    agrid = anchor_grid.get_anchor_grid(
        3,
        4, 
        8.0,
        [32, 64, 96, 128, 196],
        [0.25, 0.5, 0.75, 1.0]
    )

    dataloader = get_dataloader(train_data_dir, 224, 32, 0, agrid, False)

    i = 0

    for batch in dataloader:

        if i == 12:
            break
        images, labels, ids = batch
        img06 = images[5]
        gts = labels[5]
        id = ids[5]

        img_path = train_data_dir + f"/{id}.jpg"
        img = Image.open(img_path)
        gt_path = train_data_dir + f"/{id}.gt_data.txt"
        gts = annotation.read_groundtruth_file(gt_path)

        lgrid = label_grid.get_label_grid(anchor_grid=agrid, gts=gts, min_iou=0.7)

        label_grid.draw_matching_rects(img, agrid, lgrid)
        img.show()

        i += 1
