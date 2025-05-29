import torch
import torch.nn as nn
import torchvision

class MmpNet(torch.nn.Module):
    def __init__(self, num_widths: int, num_aspect_ratios: int):
        super().__init__()
        self.num_widths = num_widths
        self.num_aspect_ratios = num_aspect_ratios
        self.model = torchvision.models.mobilenet_v2(weights = "DEFAULT").features
        self.head = None
    
    def set_classifier_head(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.head = nn.Sequential(
            nn.Conv2d(1280, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(rows, cols), mode="bilinear", align_corners=False),
            nn.Conv2d(
                128,
                self.num_widths * self.num_aspect_ratios,
                kernel_size=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        out = self.head(features)
        # Resize 1 dim tensor to (width_sizes, ratios_sizes, rows, cols)
        # Shape is not the same due to batching
        batch_size = x.shape[0]
        out = out.reshape(batch_size, self.num_widths, self.num_aspect_ratios, self.rows, self.cols)
        return out

def step(model, criterion, optimizer, img_batch, lbl_batch) -> float:

    optimizer.zero_grad()

    prediction = model(img_batch)
    loss = criterion(prediction, lbl_batch)
    loss.backward()
    optimizer.step()

    return loss

