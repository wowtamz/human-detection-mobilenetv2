import torch


class MmpNet(torch.nn.Module):
    def __init__(self, num_widths: int, num_aspect_ratios: int):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
