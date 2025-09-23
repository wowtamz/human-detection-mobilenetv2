import torchvision

from typing import Tuple

def get_transforms(img_size: int, padding: Tuple, augmentations: list) -> list:
    
    transforms = [
        torchvision.transforms.Pad(padding, 0, "constant"),
        torchvision.transforms.Resize((img_size, img_size))
    ]
    
    transforms += augmentations

    transforms += [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    return transforms

