from typing import Sequence
from ..a3.annotation import AnnotationRect
from ..a3 import annotation
from . import anchor_grid
import numpy as np
from PIL import Image, ImageDraw


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

    return intersect_rect.area() / union_rect.area()


def get_label_grid(
    anchor_grid: np.ndarray, gts: Sequence[AnnotationRect], min_iou: float
) -> tuple[np.ndarray, ...]:
    
    sizes, ratios, rows, cols, points = anchor_grid.shape

    grid = np.ndarray((sizes, ratios, rows, cols))

    for size in range(sizes):
        for ratio in range(ratios):
            for row in range(rows):
                for col in range(cols):
                    for gt in gts:
                        rect = anchor_grid[size][ratio][row][col]
                        if iou(gt, AnnotationRect.fromarray(rect)) >= min_iou:
                            grid[size][ratio][row][col] = 1
                        else:
                            grid[size][ratio][row][col] = 1
    
    return grid

"""Exercise 4.2 (c)"""

def draw_matching_rects(img, anch_grid, label_grid):
    sizes, ratios, rows, cols, points = anch_grid.shape
    for size in range(sizes):
        for ratio in range(ratios):
            for row in range(rows):
                for col in range(cols):
                    if label_grid[size][ratio][row][col] == 1:
                        img_draw = ImageDraw.Draw(img)
                        rect = anch_grid[size][ratio][row][col]
                        annotation.draw_annotation(img_draw, rect)



def exercise_4_2_c():
    path = "/home/tamz/Documents/programming/mmp_sose25_varadita/dataset/train/"
    img_path = path + "00114403.jpg"
    gt_path = path + "00114403.gt_data.txt"
    img = Image.open(img_path)
    gts = annotation.read_groundtruth_file(gt_path)

    agrid = anchor_grid.get_anchor_grid(
        3,
        5, 
        8.0,
        [32, 64, 96, 128, 196],
        [0.25, 0.5, 0.75, 1.0, 2.0]
    )

    label_grid = get_label_grid(anchor_grid=agrid, gts=gts, min_iou=0.7)
    draw_matching_rects(img, agrid, label_grid)
    img.show()
    
if __name__ == "__main__":
    exercise_4_2_c()



