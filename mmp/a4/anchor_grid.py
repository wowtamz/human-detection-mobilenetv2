from typing import Sequence
import numpy as np

def get_anchor_grid(
    num_rows: int,
    num_cols: int,
    scale_factor: float,
    anchor_widths: Sequence[float],
    aspect_ratios: Sequence[float],
) -> np.ndarray:
    
    grid = np.zeros((len(anchor_widths), len(aspect_ratios), num_rows, num_cols, 4), dtype=np.float32)
    
    for row in range(num_rows):
        for col in range(num_cols):
            center_x = (col * 0.5) * scale_factor
            center_y = (row * 0.5) * scale_factor
            for width_idx in range(len(anchor_widths)):
                for ratio_idx in range(len(aspect_ratios)):
                    # aspect ration = height / width => height = aspect_ration * width
                    width = anchor_widths[width_idx]
                    ratio = aspect_ratios[ratio_idx]
                    height = ratio * width
                    
                    box_x1 = center_x - (width / 2)
                    box_y1 = center_y - (height / 2)
                    box_x2 = center_x + (width / 2)
                    box_y2 = center_y + (height / 2)

                    grid[width_idx][ratio_idx][row][col] = [box_x1, box_y1, box_x2, box_y2]
    return grid