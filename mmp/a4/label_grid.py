from typing import Sequence
from ..a3.annotation import AnnotationRect
from ..a3 import annotation
from . import anchor_grid
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

def iou(rect1: AnnotationRect, rect2: AnnotationRect) -> float:
    intersect_rect_x1 = max(rect1.x1, rect2.x1)
    intersect_rect_y1 = max(rect1.y1, rect2.y1)
    intersect_rect_x2 = min(rect1.x2, rect2.x2)
    intersect_rect_y2 = min(rect1.y2, rect2.y2)
    
    if (intersect_rect_x1 >= intersect_rect_x2 or intersect_rect_y1 >= intersect_rect_y2):
        return 0.0

    union_rect_x1 = min(rect1.x1, rect2.x1)
    union_rect_y1 = min(rect1.y1, rect2.y1)
    union_rect_x2 = max(rect1.x2, rect2.x2)
    union_rect_y2 = max(rect1.y2, rect2.y2)

    intersect_rect = AnnotationRect(intersect_rect_x1, intersect_rect_y1, intersect_rect_x2, intersect_rect_y2)
    union_rect = AnnotationRect(union_rect_x1, union_rect_y1, union_rect_x2, union_rect_y2)
    union_area = rect1.area() + rect2.area() - intersect_rect.area()

    return intersect_rect.area() / union_area#union_rect.area()


def get_label_grid(
    anchor_grid: np.ndarray, gts: Sequence[AnnotationRect], min_iou: float
) -> tuple[np.ndarray, ...]:
    
    sizes, ratios, rows, cols, points = anchor_grid.shape

    grid_size = sizes * ratios * rows * cols

    grid = [None] * grid_size # Preallocate grid memory
    grid = np.array(grid, dtype=bool)

    anchor_grid_flat = anchor_grid.reshape(-1, 4)

    for i in range(grid_size):
        for gt in gts:
            rect = anchor_grid_flat[i]
            if iou(AnnotationRect.fromarray(rect), gt) >= min_iou:
                grid[i] = True
                continue
    
    grid = grid.reshape((sizes, ratios, rows, cols))

    '''
    grid = np.zeros((sizes, ratios, rows, cols), dtype=bool)

    for size in range(sizes):
        for ratio in range(ratios):
            for row in range(rows):
                for col in range(cols):
                    for gt in gts:
                        rect = anchor_grid[size, ratio, row, col]
                        if iou(AnnotationRect.fromarray(rect), gt) >= min_iou:
                            grid[size, ratio, row, col] = True
    '''
    
    return grid

"""Exercise 4.2 (c)"""

def draw_matching_rects(img, anch_grid, label_grid):
    sizes, ratios, rows, cols, points = anch_grid.shape
    img_draw = ImageDraw.Draw(img)
    for size in range(sizes):
        for ratio in range(ratios):
            for row in range(rows):
                for col in range(cols):
                    if label_grid[size][ratio][row][col]:
                        rect = anch_grid[size][ratio][row][col]
                        annotation.draw_annotation(img_draw, rect, "green", 2)

def exercise_4_2_c():
    path = f"{Path.cwd()}/dataset/train/"
    img_path = path + "00114403.jpg"
    img = Image.open(img_path)

    scale_factor = 8.0
    width, height = img.size

    num_cols = int(width / scale_factor)
    num_rows = int(height / scale_factor)

    anchor_widths = [32.0, 64.0, 128.0, 256.0]
    aspect_ratios = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2]

    agrid = anchor_grid.get_anchor_grid(
        num_rows,
        num_cols, 
        scale_factor,
        anchor_widths,
        aspect_ratios
    )

    gt_path = path + "00114403.gt_data.txt"
    gts = annotation.read_groundtruth_file(gt_path)

    label_grid = get_label_grid(anchor_grid=agrid, gts=gts, min_iou=0.7)
    draw_matching_rects(img, agrid, label_grid)
    img.show()
    img.save("mmp/a4/exercise_4_2_c.png")
    
if __name__ == "__main__":
    exercise_4_2_c()
