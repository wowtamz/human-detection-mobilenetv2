import torch
import torch.nn as nn
import torchvision

class MmpNet(torch.nn.Module):
    def __init__(self, num_widths: int, num_aspect_ratios: int, rows: int = 28, cols: int = 28):
        super().__init__()
        self.num_widths = num_widths
        self.num_aspect_ratios = num_aspect_ratios
        self.rows = rows
        self.cols = cols
        self.model = torchvision.models.mobilenet_v2(weights = "DEFAULT").features # only use mobilenet's features (ex. 5.1 b)

        # custom Classifier Label Grid 4 (Page 14)
        channels_in = self.model[-1][0].out_channels # model-output: 1280 channels
        channels_out = 2 * num_widths * num_aspect_ratios
        
        self.custom = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=1),
            nn.Upsample(size=(rows, cols), mode="bilinear", align_corners=False) # Upsample to match (rows, cols)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        out = self.custom(features)
        # resize 1 dim tensor to (batch_size, 2, width_sizes, ratios_sizes, rows, cols)
        # shape is not the same due to batching
        batch_size = x.shape[0]
        out =  out.view(batch_size, 2, self.num_widths, self.num_aspect_ratios, self.rows, self.cols)
        return out
