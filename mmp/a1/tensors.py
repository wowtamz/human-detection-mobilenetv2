import torch
import torchvision
from PIL import Image

def avg_color(img: torch.Tensor):
    #r = img[0]
    #g = img[1]
    #b = img[2]
    return img.mean(dim=1).mean(dim=1) # torch.tensor([r.mean(), g.mean(), b.mean()])

def mask(foreground: torch.Tensor, background: torch.Tensor, mask_tensor: torch.Tensor, threshold: float):
    
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    
    mask_expanded = mask_tensor.expand_as(foreground)

    condition = mask_expanded > threshold
    background[condition] = foreground[condition]
    return background

def add_matrix_vector(matrix: torch.Tensor, vector: torch.Tensor):
    mat = matrix + vector
    return mat