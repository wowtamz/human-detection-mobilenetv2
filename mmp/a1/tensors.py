import torch

def avg_color(img: torch.Tensor):
    raise NotImplementedError()

def mask(foreground: torch.Tensor, background: torch.Tensor, mask_tensor: torch.Tensor, threshold: float):
    raise NotImplementedError()

def add_matrix_vector(matrix: torch.Tensor, vector: torch.Tensor):
    raise NotImplementedError()
