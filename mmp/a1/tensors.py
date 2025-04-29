import torch
import torchvision
from PIL import Image

def avg_color(img: torch.Tensor):
    #r = img[0]
    #g = img[1]
    #b = img[2]
    return img.mean(dim=1).mean(dim=1) # torch.tensor([r.mean(), g.mean(), b.mean()])

def mask(foreground: torch.Tensor, background: torch.Tensor, mask_tensor: torch.Tensor, threshold: float):
    
    condition = mask_tensor > threshold
    background[condition] = foreground
    return background

def add_matrix_vector(matrix: torch.Tensor, vector: torch.Tensor):
    mat = matrix.T + vector
    return mat