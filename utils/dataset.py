from typing import Tuple
import numpy as np
import torch
import torchvision
import os
from torch.utils.data import DataLoader
from PIL import Image

from . import annotation
from utils import label_grid


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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        @return: 3-tuple of image tensor, label grid, and image (file-)number
        """
        img_path = self.image_paths[idx]
        img_id = self.get_image_id(img_path)
        img = Image.open(img_path)
        width, height = img.size
        size_delta = abs(height - width)
        pad_x = size_delta if height > width else 0
        pad_y = size_delta if height < width else 0
        padding = (0, 0, pad_x, pad_y)

        scale = self.image_size / max(width, height)

        annotations_scaled = list(
            map(lambda a: a.scaled(scale), self.annotation_dict[img_id])
        )

        def_transforms = [
            torchvision.transforms.Pad(padding, 0, "constant"),
            torchvision.transforms.Resize((self.image_size, self.image_size)),
            # Augmentations added here
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        for augmentation in self.augmentations:
            if isinstance(augmentation, torchvision.transforms.RandomHorizontalFlip):
                for rect in annotations_scaled:
                    rect.flip_horizontal(self.image_size)

            if isinstance(augmentation, torchvision.transforms.RandomRotation):
                rotate_degrees = augmentation.degrees[0]
                annotations_scaled = [rect.rotate(rotate_degrees, self.image_size) for rect in annotations_scaled]
            
            def_transforms.insert(2, augmentation)
            
        self.annotation_dict[img_id] = annotations_scaled
        
        tfm = torchvision.transforms.Compose(def_transforms)
        img_tensor = tfm(img)
        l_grid = torch.Tensor() if len(annotations_scaled) == 0 else label_grid.get_label_grid(self.anchor_grid, annotations_scaled, self.min_iou)

        return (img_tensor, l_grid, img_id)

    def __len__(self) -> int:
        return len(self.image_paths)

# Color space transformations
def get_color_transformation(brightness=0.0, contrast=0.0, saturation=0.0):
    return torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)

def get_grayscale_transformation():
    return torchvision.transforms.Grayscale(num_output_channels=3)

def get_blur_transformation(kernel_size=(5, 5), sigma=(0.1, 5.0)):
    return torchvision.transforms.GaussianBlur(kernel_size, sigma)

# Geometric transformations
def get_horizontal_flip_transformation():
    return torchvision.transforms.RandomHorizontalFlip(1.0)

def get_rotation_transformation(degrees=45.0):
    return torchvision.transforms.RandomRotation((degrees, degrees))

def get_dataloader(
    path_to_data: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    anchor_grid: np.ndarray,
    is_test: bool,
    augmentations: list = []
) -> DataLoader:

    min_iou = 0.7
    
    dataset = MMP_Dataset(path_to_data, image_size, anchor_grid, min_iou, is_test=is_test, augmentations=augmentations)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=not is_test,
                            drop_last=not is_test)
    
    return dataloader