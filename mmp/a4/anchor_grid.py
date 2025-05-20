from typing import Sequence
import numpy as np


def get_anchor_grid(
    num_rows: int,
    num_cols: int,
    scale_factor: float,
    anchor_widths: Sequence[float],
    aspect_ratios: Sequence[float],
) -> np.ndarray:
    
    grid = np.ndarray((num_rows, num_cols, anchor_widths, aspect_ratios))

    for i in range(num_rows):
        for j in range(num_cols):
            for w in range(len(anchor_widths)):
                for r in range(len(aspect_ratios)):
                    # aspect ration = height / width => height = aspect_ration * width
                    h = r * w
                    box_center_x = (j * scale_factor) + (scale_factor / 2)
                    box_center_y = (i * scale_factor) + (scale_factor / 2)
                    box_x1 = box_center_x - (w / 2)
                    box_y1 = box_center_y - (h / 2)
                    box_x2 = box_center_x + (w / 2)
                    box_y2 = box_center_y + (h / 2)

                    grid[i][j][w][r] = np.array([box_x1, box_y1, box_x2, box_y2])