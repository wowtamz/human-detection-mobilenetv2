from typing import Tuple
from PIL import Image
from . import annotation
import os
import torch
import torchvision
from torch.utils.data import DataLoader

class MMP_Dataset(torch.utils.data.Dataset):
    """Exercise 3.2"""

    def __init__(self, path_to_data: str, image_size: int):
        """
        @param path_to_data: Path to the folder that contains the images and annotation files, e.g. dataset_mmp/train
        @param image_size: Desired image size that this dataset should return
        """
        self.path_to_data = path_to_data
        self.image_size = image_size
        self.image_paths = self.get_image_paths(path_to_data)
        self.annotation_dict = self.get_annotation_dict()
    
    def get_image_paths(self, path_to_data) -> list[str]:
        path = list(f"{path_to_data}/{p}" for p in os.listdir(path_to_data) if p.endswith(".jpg"))
        return list(sorted(path))

    def get_annotation_dict(self) -> dict:
        dictionary = dict()
        for i in self.image_paths:
            gt_path = i.replace(".jpg", ".gt_data.txt")
            if os.path.exists(gt_path):
                dictionary[i] = annotation.read_groundtruth_file(gt_path)
        return dictionary

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        @return: Tuple of image tensor and label. The label is 0 if there is one person and 1 if there a multiple people.
        """
        img_path = self.image_paths[idx]
        annotations = []
        if img_path in self.annotation_dict.keys():
            annotations = self.annotation_dict[img_path]

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

        return (img_tensor, 0 if len(annotations) <= 1 else 1)

    def __len__(self) -> int:
        return len(self.image_paths)

def get_dataloader(
        path_to_data: str, image_size: int, batch_size: int, num_workers: int, is_train: bool = True
) -> DataLoader:

    """Exercise 3.2d"""
    dataset = MMP_Dataset(path_to_data, image_size)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=is_train,
                            drop_last=is_train)
    
    return dataloader
    


