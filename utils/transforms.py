import torch
import torchvision

from typing import Tuple
from PIL import Image
from utils.annotation import AnnotationRect
from torchvision.transforms.functional import to_pil_image

def tensor_to_img(img_tensor: torch.Tensor):
    img_tensor = img_tensor.detach().cpu()

    # Denormalize image tensor
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm_tensor = img_tensor * std + mean

    img = to_pil_image(denorm_tensor)
    return img

def img_to_tensor(img, size: int, augmentations: list = []) -> torch.Tensor:
    _, padding = get_scale_and_padding(img, size)
    transforms = get_transforms(size, padding, augmentations)
    tensor = apply_transforms_to_img(img, transforms)
    return tensor

def get_scale_and_padding(img, size):
    width, height = img.size
    size_delta = abs(height - width)
    pad_x = size_delta if height > width else 0
    pad_y = size_delta if height < width else 0
    padding = (0, 0, pad_x, pad_y)
    scale = size / max(width, height)
    return scale, padding

# Color space transformations
def get_color_augment(brightness=0.0, contrast=0.0, saturation=0.0):
    return torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)

def get_grayscale_augment():
    return torchvision.transforms.Grayscale(num_output_channels=3)

def get_blur_augment(kernel_size=(5, 5), sigma=(0.1, 5.0)):
    return torchvision.transforms.GaussianBlur(kernel_size, sigma)

# Geometric transformations
def get_horizontal_flip_augment():
    return torchvision.transforms.RandomHorizontalFlip(1.0)

def get_rotation_augment(degrees=45.0):
    return torchvision.transforms.RandomRotation((degrees, degrees))

def apply_transforms_to_img(img: Image, transforms: list) -> torch.Tensor:
    compose = torchvision.transforms.Compose(transforms)
    tensor_img = compose(img)
    return tensor_img

def apply_transforms_to_annotations(annotations: list, scale: float, transforms: list, img_size: int) -> list[AnnotationRect]:
    
    annotations = get_scaled_img_annotations(annotations, scale)
   
    for transform in transforms:
        
        if isinstance(transform, torchvision.transforms.RandomHorizontalFlip):
            annotations = [rect.flip_horizontal(img_size) for rect in annotations]

        if isinstance(transform, torchvision.transforms.RandomRotation):
            rotate_degrees = transform.degrees[0]
            annotations = [rect.rotate(rotate_degrees, img_size) for rect in annotations]

    return annotations

def get_scaled_img_annotations(annotations: list, scale: float) -> list[AnnotationRect]:
    return list(
        map(lambda a: a.scaled(scale), annotations)
    )

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

