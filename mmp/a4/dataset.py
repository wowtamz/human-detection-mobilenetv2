from typing import Tuple
import numpy as np
import torch
import torchvision
import os
from torchvision.transforms.functional import to_pil_image
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
        self.path_to_data = path_to_data if path_to_data.endswith("/") else path_to_data + "/"
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
            key = i.removeprefix(self.path_to_data).removesuffix(".jpg")
            gt_path = i.replace(".jpg", ".gt_data.txt")
            if os.path.exists(gt_path):

                dictionary[key] = annotation.read_groundtruth_file(gt_path)
            else:
                dictionary[key] = []
        return dictionary

    def get_image_id(self, image_path):
        return image_path.removeprefix(self.path_to_data).removesuffix(".jpg")
    
    def get_image_path(self, image_id):
        return self.path_to_data + f"{image_id}.jpg"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        @return: 3-tuple of image tensor, label grid, and image (file-)number
        """
        img_path = self.image_paths[idx]
        img_id = self.get_image_id(img_path)
        img = Image.open(img_path)
        width, height = img.size
        size_delta = abs(height - width)
        pad_right = size_delta if height > width else 0
        pad_bottom = size_delta if height < width else 0
        padding = (0, 0, pad_right, pad_bottom)

        scale = self.image_size / max(width, height)

        annotations_scaled = list(
            map(lambda a: a.scaled(scale), self.annotation_dict[img_id])
        )
        
        tfm = torchvision.transforms.Compose([
            torchvision.transforms.Pad(padding, 0, "constant"),
            torchvision.transforms.Resize((self.image_size, self.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = tfm(img)
        l_grid = torch.Tensor() if len(annotations_scaled) == 0 else label_grid.get_label_grid(self.anchor_grid, annotations_scaled, self.min_iou)

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

    min_iou = 0.7
    
    dataset = MMP_Dataset(path_to_data, image_size, anchor_grid, min_iou, is_test=is_test)

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
    loader.dataset.min_iou = min_iou

    gt_count = 0
    covered_count = 0

    for i in range(len(loader.dataset)):
        tup = loader.dataset[i] # (img_tensor, lgrid, img_id)
        lgrid = tup[1]
        img_id = tup[2]

        gt_count += len(loader.dataset.annotation_dict[img_id])
        covered_count += np.count_nonzero(lgrid == 1)
    
    return covered_count / gt_count

# For comparing scaled image and its gts to the original image
def draw_original(img_path):

    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    gt_path = img_path.replace(".jpg", ".gt_data.txt")
    gts = annotation.read_groundtruth_file(gt_path)

    for gt in gts:
        annotation.draw_annotation(draw, gt.__array__())
    
    img.show()

if __name__ == "__main__":
    train_data_dir = f"{Path.cwd()}/dataset/train"

    img_size = 224
    batch_size = 8
    num_workers = 0

    scale_factor = 4.0

    num_rows = int(img_size / scale_factor)
    num_cols = int(img_size / scale_factor)

    anchor_widths = [2, 4, 8, 16, 32, 64, 128, 256]
    aspect_ratios = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3]

    agrid = anchor_grid.get_anchor_grid(
        num_rows,
        num_cols, 
        scale_factor,
        anchor_widths,
        aspect_ratios
    )

    dataloader = get_dataloader(train_data_dir, img_size, batch_size, num_workers, agrid, is_test=True)
    dataloader.dataset.min_iou = 0.6
    #max_coverage = calculate_max_coverage(dataloader, 0.5)
    #print("<=== Max Coverage ===>")
    #print(max_coverage)

    batch_count = 12
    current_batch = 0

    for batch in dataloader:

        if current_batch == batch_count:
            break
        
        n = 5
        images, labels, ids = batch
        img06_tensor = images[n]
        lgrid = labels[n]
        img_id = ids[n]

        # Denormalize image tensor
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denorm_tensor = img06_tensor * std + mean

        img = to_pil_image(denorm_tensor)

        if lgrid.numel() != 0:
            label_grid.draw_matching_rects(img, agrid, lgrid)
        img.show()
        img.save(f"mmp/a4/img06-batch{current_batch}.jpg")

        current_batch += 1